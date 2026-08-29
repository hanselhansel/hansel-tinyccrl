.PHONY: all engine web clean

all: engine web

engine:
	cargo build --release -p tinyccrl-engine

web:
	cd engine && wasm-pack build --target web --out-dir ../web/public/pkg --features wasm
	cd web && npm install && npm run build

clean:
	cargo clean
	rm -rf web/dist web/public/pkg
