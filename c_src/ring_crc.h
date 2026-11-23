#ifndef PAR_RING_CRC_H
#define PAR_RING_CRC_H

#include <stddef.h>
#include <stdint.h>

#define PAR_RING_MAGIC       0x50415231U
#define PAR_RING_VERSION     1U
#define PAR_MAX_PAYLOAD_LEN  192U
#define PAR_CACHE_LINE_BYTES 64U
#define PAR_HDR_HASH_BYTES   40U

#define PAR_BUCKET_MAGIC     0x50544231U
#define PAR_BUCKET_VERSION   1U
#define PAR_NS_PER_SEC       1000000000ULL

#define PAR_OK                0
#define PAR_ERR_NULL         (-1)
#define PAR_ERR_ALIGN        (-2)
#define PAR_ERR_BOUNDS       (-3)
#define PAR_ERR_MAGIC        (-4)
#define PAR_ERR_VERSION      (-5)
#define PAR_ERR_RESERVED     (-6)
#define PAR_ERR_HDR_CRC      (-7)
#define PAR_ERR_PAYLOAD_CRC  (-8)
#define PAR_ERR_RATE_PARAM   (-9)

typedef struct ParRingSlotHeader {
    _Alignas(64) uint32_t magic;
    uint16_t version;
    uint16_t flags;
    uint64_t sequence;
    uint64_t timestamp_ns;
    uint32_t agent_id;
    uint32_t capability_mask;
    uint32_t payload_len;
    uint32_t payload_crc32c;
    uint32_t header_crc32c;
    uint8_t reserved[20];
} ParRingSlotHeader;

typedef struct ParTokenBucket {
    _Alignas(64) uint32_t magic;
    uint16_t version;
    uint16_t flags;
    uint64_t capacity_milli;
    uint64_t tokens_milli;
    uint64_t refill_per_sec_milli;
    uint64_t last_refill_ns;
    uint64_t rem_ns_num;
    uint64_t allowed_count;
    uint64_t denied_count;
} ParTokenBucket;

uint32_t par_crc32c(const uint8_t *data, size_t len);

int32_t par_ring_header_seal(
    ParRingSlotHeader *hdr,
    const uint8_t *payload,
    size_t payload_len
);

int32_t par_ring_header_validate(
    const ParRingSlotHeader *hdr,
    const uint8_t *payload,
    size_t payload_len
);

int32_t par_token_bucket_init(
    ParTokenBucket *tb,
    uint64_t capacity_milli,
    uint64_t refill_per_sec_milli,
    uint64_t initial_ns
);

int32_t par_token_bucket_try_acquire(
    ParTokenBucket *tb,
    uint64_t cost_milli,
    uint64_t now_ns,
    uint64_t *retry_after_ns_out
);

#endif /* PAR_RING_CRC_H */
