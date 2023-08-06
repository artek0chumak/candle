clean-ptx:
	find target -name "*.ptx" -type f -delete
	echo "" > candle-kernels/src/lib.rs
	touch candle-kernels/build.rs

clean:
	cargo clean

test:
	cargo test

pyo3-test:
	cargo build --profile=release-with-debug --package candle-pyo3
	ln -f -s ./target/release-with-debug/libcandle.so candle.so
	PYTHONPATH=. python3 candle-pyo3/test.py

all: test
