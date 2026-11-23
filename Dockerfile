FROM rust:1.83-slim-bookworm AS builder
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY Cargo.toml build.rs ./
COPY c_src ./c_src
COPY src ./src
RUN cargo test --release --verbose && cargo build --release --bin bench_runtime

FROM debian:bookworm-slim
RUN useradd --create-home --uid 10001 appuser
USER appuser
WORKDIR /home/appuser
COPY --from=builder /workspace/target/release/bench_runtime /usr/local/bin/bench_runtime
ENTRYPOINT ["/usr/local/bin/bench_runtime"]
