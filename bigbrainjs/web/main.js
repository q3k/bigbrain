import { Brain, default as init } from './pkg/bigbrainjs.js';

let canvas = null;
let ctx = null;
let pos = { x: 0, y: 0 };
let brain = null;

document.addEventListener('DOMContentLoaded', async () => {
    await init();
    let response = await fetch('/net.pb');
    let blob = await response.blob(); // download as Blob object
    let data = new Uint8Array(await blob.arrayBuffer());
    brain = Brain.new(data);
    console.log(brain);

    canvas = document.getElementById("myCanvas");
    ctx = canvas.getContext('2d');
    const w = canvas.getBoundingClientRect().width;
    const h = canvas.getBoundingClientRect().height;
    ctx.rect(0, 0, w, h);
    ctx.fillStyle = "white";
    ctx.fill();
});

const pokeBrain = () => {
    const w = canvas.getBoundingClientRect().width;
    const h = canvas.getBoundingClientRect().height;
    let data = ctx.getImageData(0, 0, w, h);

    let sx = w / 28;
    let sy = h / 28;

    let mnist = new Uint8Array(28*28);
    for (let x = 0; x < 28; x++) {
        for (let y = 0; y < 28; y++) {
            let cx = Math.floor(x * sx);
            let cy = Math.floor(y * sy);
            let ix = (cx + cy * w) * 4;
            //let ix = x + y * w * 4;
            //console.log(data.data[Math.floor(x*sx + y*sy*w)]);
            //console.log(data.data[ix]);
            mnist[x+y*28] = data.data[ix];
        }
    }

    //for (let y = 0; y < 28; y++) {
    //    let line = Array.from(mnist.subarray(y*28, (y+1)*28)).map((el) => {
    //        if (el == 0) {
    //            return "#";
    //        }
    //        return ".";
    //    });
    //    console.log(y, line.join(""));
    //}
    //console.log("===");

    let res = brain.see(mnist);
    //console.log(res);
    let i = 0;
    for (let digit of res) {
        let val = digit * 100 + 20 + "px";
        document.querySelector("#digit"+i).style.height = val;
        i += 1;
    }
};

document.addEventListener('mousemove', (e) => {
    if (e.buttons !== 1) return;
  
    ctx.beginPath(); // begin
  
    ctx.lineWidth = 16;
    ctx.lineCap = 'round';
    ctx.strokeStyle = '#000000';
  
    ctx.moveTo(pos.x, pos.y); // from
    pos.x = e.clientX;
    pos.y = e.clientY;
    ctx.lineTo(pos.x, pos.y); // to
  
    ctx.stroke(); // draw it!

    pokeBrain();
});
document.addEventListener('mousedown', (e) => {
    pos.x = e.clientX;
    pos.y = e.clientY;
});
document.addEventListener('mouseenter', (e) => {
    pos.x = e.clientX;
    pos.y = e.clientY;
});
