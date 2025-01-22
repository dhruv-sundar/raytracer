TARGET = raytracer

all: build

build:
	cargo build --release

run: build
	cargo run $(file)

clean:
	cargo clean

run_fast: build_fast
	cargo run $(file)

build_fast:
	cargo build --release

	