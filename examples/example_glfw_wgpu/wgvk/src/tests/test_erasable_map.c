#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>


#include <wgvk.h>
#include <wgvk_structs_impl.h>

DEFINE_PTR_HASH_MAP_ERASABLE(static, ErasableIntMap, int)

static int g_test_failures = 0;
#define TEST_ASSERT(condition) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "TEST FAILED: %s at %s:%d\n", #condition, __FILE__, __LINE__); \
            g_test_failures++; \
        } \
    } while (0)

void test_initialization() {
    printf("--- Running test_initialization ---\n");
    ErasableIntMap map;
    ErasableIntMap_init(&map);
    TEST_ASSERT(map.current_size == 0);
    TEST_ASSERT(map.tombstone_count == 0);
    TEST_ASSERT(map.current_capacity == 0);
    TEST_ASSERT(map.has_null_key == false);
    TEST_ASSERT(map.table == NULL);
    TEST_ASSERT(ErasableIntMap_get(&map, (void*)0x100) == NULL);
    ErasableIntMap_free(&map);
}

void test_put_and_get() {
    printf("--- Running test_put_and_get ---\n");
    ErasableIntMap map;
    ErasableIntMap_init(&map);

    int key1 = 100, key2 = 200, key3 = 300;
    
    // Test basic insertion
    TEST_ASSERT(ErasableIntMap_put(&map, &key1, 10) == 1); // New
    TEST_ASSERT(map.current_size == 1);
    TEST_ASSERT(*ErasableIntMap_get(&map, &key1) == 10);

    // Test update
    TEST_ASSERT(ErasableIntMap_put(&map, &key1, 11) == 0); // Update
    TEST_ASSERT(map.current_size == 1);
    TEST_ASSERT(*ErasableIntMap_get(&map, &key1) == 11);

    // Test NULL key
    TEST_ASSERT(ErasableIntMap_put(&map, NULL, 99) == 1); // New
    TEST_ASSERT(map.has_null_key == true);
    TEST_ASSERT(*ErasableIntMap_get(&map, NULL) == 99);
    TEST_ASSERT(ErasableIntMap_put(&map, NULL, 101) == 0); // Update
    TEST_ASSERT(*ErasableIntMap_get(&map, NULL) == 101);

    // Test getting a non-existent key
    TEST_ASSERT(ErasableIntMap_get(&map, &key2) == NULL);

    ErasableIntMap_free(&map);
}

void test_erase() {
    printf("--- Running test_erase ---\n");
    ErasableIntMap map;
    ErasableIntMap_init(&map);

    int key1 = 100, key2 = 200;

    ErasableIntMap_put(&map, &key1, 10);
    ErasableIntMap_put(&map, &key2, 20);
    ErasableIntMap_put(&map, NULL, 99);

    TEST_ASSERT(map.current_size == 2);
    TEST_ASSERT(map.has_null_key == true);

    // Erase a non-existent key
    TEST_ASSERT(ErasableIntMap_erase(&map, (void*)0xDEAD) == false);
    TEST_ASSERT(map.current_size == 2);
    TEST_ASSERT(map.tombstone_count == 0);

    // Erase an existing key
    TEST_ASSERT(ErasableIntMap_erase(&map, &key1) == true);
    TEST_ASSERT(map.current_size == 1);
    TEST_ASSERT(map.tombstone_count == 1);
    TEST_ASSERT(ErasableIntMap_get(&map, &key1) == NULL);
    TEST_ASSERT(*ErasableIntMap_get(&map, &key2) == 20); // Make sure other keys are findable

    // Erase NULL key
    TEST_ASSERT(ErasableIntMap_erase(&map, NULL) == true);
    TEST_ASSERT(map.has_null_key == false);
    TEST_ASSERT(ErasableIntMap_get(&map, NULL) == NULL);

    // Erase the same key again
    TEST_ASSERT(ErasableIntMap_erase(&map, &key1) == false);

    ErasableIntMap_free(&map);
}

void test_growth_and_rehash() {
    printf("--- Running test_growth_and_rehash ---\n");
    ErasableIntMap map;
    ErasableIntMap_init(&map);

    // Use an array of keys to avoid stack addresses changing
    #define NUM_KEYS 20
    int keys[NUM_KEYS];
    for (int i = 0; i < NUM_KEYS; i++) keys[i] = i;

    uint64_t initial_cap = 0;
    
    for (int i = 0; i < NUM_KEYS; i++) {
        if (map.current_capacity != initial_cap) {
            printf("Map grew from %llu to %llu at insertion %d\n", (unsigned long long)initial_cap, (unsigned long long)map.current_capacity, i);
            initial_cap = map.current_capacity;
        }
        ErasableIntMap_put(&map, &keys[i], i * 10);
    }

    TEST_ASSERT(map.current_size == NUM_KEYS);
    TEST_ASSERT(map.current_capacity >= NUM_KEYS);

    // Verify all keys are present after growth
    for (int i = 0; i < NUM_KEYS; i++) {
        int* val = ErasableIntMap_get(&map, &keys[i]);
        TEST_ASSERT(val != NULL && *val == i * 10);
    }

    // Erase some keys
    for (int i = 0; i < NUM_KEYS / 2; i++) {
        ErasableIntMap_erase(&map, &keys[i * 2]); // Erase even keys
    }

    TEST_ASSERT(map.current_size == NUM_KEYS / 2);
    TEST_ASSERT(map.tombstone_count == NUM_KEYS / 2);

    // Verify correct keys are gone and others remain
    for (int i = 0; i < NUM_KEYS; i++) {
        int* val = ErasableIntMap_get(&map, &keys[i]);
        if (i % 2 == 0) {
            TEST_ASSERT(val == NULL);
        } else {
            TEST_ASSERT(val != NULL && *val == i * 10);
        }
    }

    // Add a new key, which might reuse a tombstone slot
    int new_key = 999;
    ErasableIntMap_put(&map, &new_key, 9990);
    TEST_ASSERT(map.current_size == (NUM_KEYS / 2) + 1);
    // Tombstone count might decrease if one was reused
    TEST_ASSERT(map.tombstone_count <= NUM_KEYS / 2); 
    TEST_ASSERT(*ErasableIntMap_get(&map, &new_key) == 9990);


    // Trigger another grow by adding more keys, this should clear tombstones
    uint64_t tombstone_count_before_grow = map.tombstone_count;
    TEST_ASSERT(tombstone_count_before_grow > 0);
    int more_keys[NUM_KEYS];
    for (int i = 0; i < NUM_KEYS; i++) {
        more_keys[i] = 1000 + i;
        ErasableIntMap_put(&map, &more_keys[i], 10000 + i);
    }
    
    printf("Tombstones before final grow: %llu. After grow: %llu\n", (unsigned long long)tombstone_count_before_grow, (unsigned long long)map.tombstone_count);
    TEST_ASSERT(map.tombstone_count == 0); // Grow should have cleared all tombstones

    ErasableIntMap_free(&map);
}

void test_collision_and_erase() {
    printf("--- Running test_collision_and_erase ---\n");
    ErasableIntMap map;
    ErasableIntMap_init(&map);
    
    // Force a collision. With a capacity of 8, keys that hash to the same
    // value modulo 8 will collide. E.g., pointers that differ by 8 * sizeof(int).
    // Note: This is not a guaranteed collision due to the multiplier, but it's likely.
    // A more reliable way is to find keys that actually collide.
    int key1 = 1;
    int key_collide = 1 + PHM_INITIAL_HEAP_CAPACITY; // A likely candidate for collision
    int key_after = 2;

    // To be certain, let's craft a collision.
    // cap_mask = 7 (0b111). We need Name_hash_key(k1) & 7 == Name_hash_key(k2) & 7
    void *k1 = (void*)8; // hash & 7 == 0
    void *k2 = (void*)16; // hash & 7 == 0
    void *k3 = (void*)24; // hash & 7 == 0

    // Put k1, it goes into its ideal slot.
    ErasableIntMap_put(&map, k1, 10);
    TEST_ASSERT(map.current_size == 1);
    TEST_ASSERT(map.current_capacity == PHM_INITIAL_HEAP_CAPACITY);
    
    // Put k2, it will probe past k1.
    ErasableIntMap_put(&map, k2, 20);
    TEST_ASSERT(map.current_size == 2);
    
    // Put k3, it probes past k1 and k2.
    ErasableIntMap_put(&map, k3, 30);
    TEST_ASSERT(map.current_size == 3);

    // All should be retrievable
    TEST_ASSERT(*ErasableIntMap_get(&map, k1) == 10);
    TEST_ASSERT(*ErasableIntMap_get(&map, k2) == 20);
    TEST_ASSERT(*ErasableIntMap_get(&map, k3) == 30);

    // Now, erase the key in the middle of the probe chain (k2).
    ErasableIntMap_erase(&map, k2);
    TEST_ASSERT(map.current_size == 2);
    TEST_ASSERT(map.tombstone_count == 1);
    TEST_ASSERT(ErasableIntMap_get(&map, k2) == NULL);

    // CRITICAL: Ensure that k3 is still findable, proving that the search
    // correctly skipped over the tombstone left by k2.
    TEST_ASSERT(*ErasableIntMap_get(&map, k3) == 30);

    // Now add k2 back. It should reuse the tombstone slot.
    ErasableIntMap_put(&map, k2, 22);
    TEST_ASSERT(map.current_size == 3);
    TEST_ASSERT(map.tombstone_count == 0); // Tombstone was reused
    TEST_ASSERT(*ErasableIntMap_get(&map, k2) == 22);

    ErasableIntMap_free(&map);
}


int main() {
    test_initialization();
    test_put_and_get();
    test_erase();
    test_growth_and_rehash();
    test_collision_and_erase();

    if (g_test_failures == 0) {
        printf("\nAll tests passed!\n");
        return 0;
    } else {
        printf("\n%d test(s) failed.\n", g_test_failures);
        return 1;
    }
}
