bigbrain
===

An experiment in following
[neuralnetworksanddeeplearning.com](http://neuralnetworksanddeeplearning.com/)
in Rust. The end result is a web page that can sometimes recognize hand-written
digits using a three-layer (one hidden) neural network.

![A digit '3' being written in a JS canvas, with the probability of the output being any of the digits 0-9 displayed on a bar graph underneath it.](https://object.ceph-eu.hswaw.net/q3k-personal/a61716bb765cbe53955829fcdb1d9b6d4e2b5c5a4b87b6312940bfae8a2e3a53.png)

You can see it in action at [https://hackerspace.pl/~q3k/bigbrain/main.html](https://hackerspace.pl/~q3k/bigbrain/main.html). It's really good at figuring out the digit 2, and gets somewhat confused about other digits. Be kind to it.

Goals
---

1. Be a weekend project
2. Implement everything specific to ML/DL/NN from scratch
3. End up with a janky JS/WASM demo

Non-Goals
---

1. Be fast
2. Be good
3. Be clean code

Training
---

First, acquire the MNIST handwritten digit database (training/test sets, both images and labels) and save them in this repo.

Then, `cargo run --release` to run training, which will generate a `net.pb` containing the trained model.

Building web app
---

You'll need wasm-pack (`cargo install wasm-pack`). Then:

```
    cd bigbrainjs
    wasm-pack build --release --target web
```

You can then serve files from the `bigbrainjs/web` repository to see the web interface. The simplest way to do that is probably to run `python -m http.server`.


License
===

Copyright © 2021-2023 Serge Bazanski <q3k@q3k.org>

This work is free. You can redistribute it and/or modify it under the
terms of the Do What The Fuck You Want To Public License, Version 2,
as published by Sam Hocevar. See the COPYING file for more details.
