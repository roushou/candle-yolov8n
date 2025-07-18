# Candle YoloV8n

This is a computer vision project that detects objects from your webcam using YoloV8n model in safetensors format.

It is built in Rust using [opencv](https://github.com/twistedfall/opencv-rust) and [candle](https://github.com/huggingface/candle) crate from [HuggingFace](https://huggingface.co/).

Inference is done on the GPU using Metal, and post-processing is done on the CPU.

## Getting started

You may need to set this environment variable on MacOS. Checkout this [page](https://github.com/twistedfall/opencv-rust?tab=readme-ov-file#environment-variables) to see instructions for your OS.

```bash
export DYLD_FALLBACK_LIBRARY_PATH="$(xcode-select --print-path)/Toolchains/XcodeDefault.xctoolchain/usr/lib/"
```

Run the app.

```bash
cargo run
```

## License

This project is licensed under the [MIT License](./LICENSE)
