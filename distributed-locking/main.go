package main

import (
	"context"
	crand "crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"log"
	mrand "math/rand"
	"sync"
	"sync/atomic"
	"time"

	"github.com/redis/go-redis/v9"
)

var ErrNotAcquired = errors.New("redlock: could not acquire lock on a quorum of nodes")

const unlockScript = `
	if redis.call("get",KEYS[1]) == ARGV[1] then
		return redis.call("del",KEYS[1])
	else
		return 0
	`

type Redlock struct {
	clients     []*redis.Client
	quorum      int           // N/2 + 1
	retries     int           // how many full acquisition attempts
	retryDelay  time.Duration // max random jitter between attempts
	nodeTimeout time.Duration // per-node timeout, must be tiny vs the lock TTL
	driftFactor float64       // clock drift allowance, 1% as in the paper
	unlock      *redis.Script
}

type Lock struct {
	Resource   string
	Token      string
	Validity   time.Duration
	ValidUntil time.Time
}

func New(addrs []string) *Redlock {
	clients := make([]*redis.Client, len(addrs))
	for i, addr := range addrs {
		clients[i] = redis.NewClient(&redis.Options{
			Addr:         addr,
			DialTimeout:  100 * time.Millisecond,
			ReadTimeout:  100 * time.Millisecond,
			WriteTimeout: 100 * time.Millisecond,
			MaxRetries:   -1, // disable go-redis internal retries, they would wreck our timing math
		})
	}
	return &Redlock{
		clients:     clients,
		quorum:      len(clients)/2 + 1,
		retries:     3,
		retryDelay:  200 * time.Millisecond,
		nodeTimeout: 50 * time.Millisecond,
		driftFactor: 0.01,
		unlock:      redis.NewScript(unlockScript),
	}
}

func (r *Redlock) Close() {
	for _, c := range r.clients {
		c.Close()
	}
}

// Acquire implements the Redlock algorithm:
//  1. take the start time
//  2. try SET NX PX on all N nodes (in parallel, tiny per-node timeout)
//  3. elapsed = now - start; validity = ttl - elapsed - drift
//  4. success only if acquired on >= N/2+1 nodes AND validity > 0
//  5. on failure, unlock on ALL nodes (even ones we think we failed on,
//     because the reply may have been lost after the write happened)
func (r *Redlock) Acquire(ctx context.Context, resource string, ttl time.Duration) (*Lock, error) {
	for attempt := range r.retries {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		token, err := getToken()
		if err != nil {
			return nil, err
		}

		// [1] start timer
		start := time.Now()

		// [2] lock acquisition on every node
		acquired := r.forEachNode(ctx, func(nctx context.Context, c *redis.Client) bool {
			ok, err := c.SetNX(nctx, resource, token, ttl).Result()
			return err == nil && ok
		})

		// [3] elapsed time and remaining validity
		elapsed := time.Since(start)
		drift := time.Duration(float64(ttl)*r.driftFactor) + 2*time.Millisecond
		validity := ttl - elapsed - drift

		if acquired >= r.quorum && validity > 0 {
			return &Lock{
				Resource:   resource,
				Token:      token,
				Validity:   validity,
				ValidUntil: start.Add(ttl - drift),
			}, nil
		}

		// [5] failed: release on ALL nodes, since a reply may have been lost
		// after the write actually happened.
		r.release(ctx, resource, token)

		if attempt < r.retries-1 && r.retryDelay > 0 {
			// Back off for a random amount up to retryDelay before retrying.
			delay := time.Duration(mrand.Int63n(int64(r.retryDelay)))
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(delay):
			}
		}
	}

	return nil, ErrNotAcquired
}

// forEachNode runs fn against every node concurrently, bounded by nodeTimeout,
// and returns how many nodes reported success.
func (r *Redlock) forEachNode(ctx context.Context, fn func(context.Context, *redis.Client) bool) int {
	var (
		wg sync.WaitGroup
		ok atomic.Int32
	)

	for _, c := range r.clients {
		wg.Go(func() {
			nctx, cancel := context.WithTimeout(ctx, r.nodeTimeout)
			defer cancel()

			if fn(nctx, c) {
				ok.Add(1)
			}
		})
	}

	wg.Wait()
	return int(ok.Load())
}

// Release unlocks on all N nodes. Safe to call even if some nodes are down:
// their copies expire by TTL anyway.
func (r *Redlock) Release(ctx context.Context, l *Lock) {
	r.release(ctx, l.Resource, l.Token)
}

func (r *Redlock) release(ctx context.Context, resource, token string) {
	r.forEachNode(ctx, func(nctx context.Context, c *redis.Client) bool {
		n, err := r.unlock.Run(nctx, c, []string{resource}, token).Int()
		return err == nil && n == 1
	})
}

func getToken() (string, error) {
	b := make([]byte, 16)
	if _, err := crand.Read(b); err != nil {
		return "", err
	}
	return hex.EncodeToString(b), nil
}

func main() {
	// N independent Redis instances, e.g. 5 containers on different ports.
	rl := New([]string{
		"localhost:6379", "localhost:6380", "localhost:6381",
		"localhost:6382", "localhost:6383",
	})
	defer rl.Close()

	ctx := context.Background()

	lock, err := rl.Acquire(ctx, "my_lock", 10*time.Second)
	if err != nil {
		log.Fatalf("failed to acquire lock: %v", err)
	}
	fmt.Printf("lock acquired, safe to work for %v\n", lock.Validity)

	// Critical section. Stop (or abort) once time.Now() passes lock.ValidUntil.
	time.Sleep(2 * time.Second)

	rl.Release(ctx, lock)
	fmt.Println("lock released")
}
