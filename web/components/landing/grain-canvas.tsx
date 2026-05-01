"use client";

import { useEffect, useRef } from "react";

const VERTEX_SHADER = `#version 300 es
precision highp float;

in vec2 position;
in vec2 uv;

out vec2 vUv;

void main() {
  vUv = uv;
  gl_Position = vec4(position, 0.0, 1.0);
}
`;

const FRAGMENT_SHADER = `#version 300 es
precision highp float;

uniform sampler2D tMap;
uniform sampler2D tMask;
uniform vec2 uScreen;
uniform vec2 uTextureSize;
uniform float uTime;
uniform vec2 uMouse;
uniform float uZoom;

in vec2 vUv;
out vec4 fragColor;

float random(float x) {
  return fract(sin(x) * 10000.0);
}

float noise(vec2 p) {
  return random(p.x + p.y * 10000.0);
}

vec2 sw(vec2 p) { return vec2(floor(p.x), floor(p.y)); }
vec2 se(vec2 p) { return vec2(ceil(p.x), floor(p.y)); }
vec2 nw(vec2 p) { return vec2(floor(p.x), ceil(p.y)); }
vec2 ne(vec2 p) { return vec2(ceil(p.x), ceil(p.y)); }

float smoothNoise(vec2 p) {
  vec2 interp = smoothstep(0.0, 1.0, fract(p));
  float s = mix(noise(sw(p)), noise(se(p)), interp.x);
  float n = mix(noise(nw(p)), noise(ne(p)), interp.x);
  return mix(s, n, interp.y);
}

float fractalNoise(vec2 p) {
  float n = 0.0;
  n += smoothNoise(p);
  n += smoothNoise(p * 2.0) / 2.0;
  n += smoothNoise(p * 4.0) / 4.0;
  n += smoothNoise(p * 8.0) / 8.0;
  n += smoothNoise(p * 16.0) / 16.0;
  n /= 1.0 + 1.0/2.0 + 1.0/4.0 + 1.0/8.0 + 1.0/16.0;
  return n;
}

vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec2 mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec3 permute(vec3 x) { return mod289(((x * 34.0) + 1.0) * x); }

float snoise(vec2 v) {
  const vec4 C = vec4(
    0.211324865405187,
    0.366025403784439,
    -0.577350269189626,
    0.024390243902439
  );
  vec2 i  = floor(v + dot(v, C.yy));
  vec2 x0 = v - i + dot(i, C.xx);
  vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;
  i = mod289(i);
  vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));
  vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), 0.0);
  m = m * m;
  m = m * m;
  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;
  m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);
  vec3 g;
  g.x = a0.x * x0.x + h.x * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}

float cubicPulse(float c, float w, float x) {
  x = abs(x - c);
  if (x > w) return 0.0;
  x /= w;
  return 1.0 - x * x * (3.0 - 2.0 * x);
}

float luma(vec3 color) {
  return dot(color, vec3(0.299, 0.587, 0.114));
}

float luma(vec4 color) {
  return dot(color.rgb, vec3(0.299, 0.587, 0.114));
}

float dither8x8(vec2 position, float brightness) {
  int x = int(mod(position.x, 8.0));
  int y = int(mod(position.y, 8.0));
  int index = x + y * 8;
  float limit = 0.0;

  if (index == 0) limit = 0.015625;
  if (index == 1) limit = 0.515625;
  if (index == 2) limit = 0.140625;
  if (index == 3) limit = 0.640625;
  if (index == 4) limit = 0.046875;
  if (index == 5) limit = 0.546875;
  if (index == 6) limit = 0.171875;
  if (index == 7) limit = 0.671875;
  if (index == 8) limit = 0.765625;
  if (index == 9) limit = 0.265625;
  if (index == 10) limit = 0.890625;
  if (index == 11) limit = 0.390625;
  if (index == 12) limit = 0.796875;
  if (index == 13) limit = 0.296875;
  if (index == 14) limit = 0.921875;
  if (index == 15) limit = 0.421875;
  if (index == 16) limit = 0.203125;
  if (index == 17) limit = 0.703125;
  if (index == 18) limit = 0.078125;
  if (index == 19) limit = 0.578125;
  if (index == 20) limit = 0.234375;
  if (index == 21) limit = 0.734375;
  if (index == 22) limit = 0.109375;
  if (index == 23) limit = 0.609375;
  if (index == 24) limit = 0.953125;
  if (index == 25) limit = 0.453125;
  if (index == 26) limit = 0.828125;
  if (index == 27) limit = 0.328125;
  if (index == 28) limit = 0.984375;
  if (index == 29) limit = 0.484375;
  if (index == 30) limit = 0.859375;
  if (index == 31) limit = 0.359375;
  if (index == 32) limit = 0.0625;
  if (index == 33) limit = 0.5625;
  if (index == 34) limit = 0.1875;
  if (index == 35) limit = 0.6875;
  if (index == 36) limit = 0.03125;
  if (index == 37) limit = 0.53125;
  if (index == 38) limit = 0.15625;
  if (index == 39) limit = 0.65625;
  if (index == 40) limit = 0.8125;
  if (index == 41) limit = 0.3125;
  if (index == 42) limit = 0.9375;
  if (index == 43) limit = 0.4375;
  if (index == 44) limit = 0.78125;
  if (index == 45) limit = 0.28125;
  if (index == 46) limit = 0.90625;
  if (index == 47) limit = 0.40625;
  if (index == 48) limit = 0.25;
  if (index == 49) limit = 0.75;
  if (index == 50) limit = 0.125;
  if (index == 51) limit = 0.625;
  if (index == 52) limit = 0.21875;
  if (index == 53) limit = 0.71875;
  if (index == 54) limit = 0.09375;
  if (index == 55) limit = 0.59375;
  if (index == 56) limit = 1.0;
  if (index == 57) limit = 0.5;
  if (index == 58) limit = 0.875;
  if (index == 59) limit = 0.375;
  if (index == 60) limit = 0.96875;
  if (index == 61) limit = 0.46875;
  if (index == 62) limit = 0.84375;
  if (index == 63) limit = 0.34375;

  return brightness < limit ? 0.0 : 1.0;
}

vec4 dither8x8(vec2 position, vec4 color) {
  return vec4(color.rgb * dither8x8(position, luma(color)), 1.0);
}

float RGB2Gray(vec3 color) {
  return 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;
}

float circle(vec2 uv, vec2 center, float radius, float feather) {
  float dist = length(uv - center);
  return smoothstep(radius, radius + feather, dist);
}

vec3 colorGradient(float t) {
  vec3 a = vec3(1.0);
  vec3 b = vec3(1.0);
  vec3 c = vec3(1.0);
  vec3 d = vec3(1.0);
  return a + b * cos(6.28318 * (c * t + d));
}

void main() {
  vec2 uv_c = vUv;
  float aspectRatio = uScreen.x / uScreen.y;
  uv_c.x *= aspectRatio;

  vec2 mouse = uMouse + vec2(0.5, 0.5);
  mouse.x *= aspectRatio;

  float baseRadius = 0.0004 + sin(uTime * 0.005) * 0.001;
  baseRadius *= uZoom;
  float dist = length(uv_c - mouse);
  float mouseInfluence = smoothstep(0.8, 0.0, dist);
  float radius = baseRadius * (1.0 + mouseInfluence * 0.5);

  float c1 = 1.0 - circle(uv_c, mouse, radius, 0.07);
  vec3 color1 = colorGradient(uv_c.x * 0.15 * 0.0001);
  vec3 bgColor = vec3(0.0);
  vec3 circleColor = bgColor;
  circleColor = mix(circleColor, color1, c1 * 0.58);

  float ripple = sin(dist * 300.0 - uTime * 0.003) * 0.04 * mouseInfluence;
  circleColor += ripple;
  circleColor *= uZoom;

  float timer = (1.0 + sin(uTime * 0.0004)) * 0.5;
  float timermid = sin(uTime * 0.0004);

  // cover-crop: scale UVs so the texture fills the viewport without distortion
  vec2 screenAspect = vec2(uScreen.x / uScreen.y, 1.0);
  vec2 texAspect = vec2(uTextureSize.x / uTextureSize.y, 1.0);
  vec2 coverScale = vec2(
    min(screenAspect.x / texAspect.x, 1.0),
    min(texAspect.x / screenAspect.x, 1.0)
  );
  vec2 uvim = vec2(vUv.x, 1.0 - vUv.y);
  uvim = uvim * coverScale + (1.0 - coverScale) * 0.5;
  uvim.y += timermid * 0.02;

  vec2 normalizedPixelSize = 4.0 / uScreen;
  vec2 uvPixel = normalizedPixelSize * floor(uvim / normalizedPixelSize);

  vec2 uvNoise = uvPixel;
  uvNoise.x *= uScreen.x / uScreen.y;
  vec3 colorNoise = vec3(0.0);
  vec2 posNoise = uvNoise * 3.0;

  float DF = 0.0;
  float a = 0.0;
  vec2 velNoise = vec2(uTime * 0.0001);
  DF += snoise(posNoise + velNoise) * 0.25 + 0.25;

  a = snoise(posNoise * vec2(cos(uTime * 0.00015), sin(uTime * 0.0001)) * 0.1) * 3.1415;
  velNoise = vec2(cos(a), sin(a));
  DF += snoise(posNoise + velNoise) * 0.25 + 0.25;

  colorNoise = 1.0 - vec3(smoothstep(0.85, 0.9, fract(DF)));

  vec4 newNoise = dither8x8(
    vec2(gl_FragCoord.x * 0.36, gl_FragCoord.y * 0.36),
    vec4(circleColor * 0.9, 1.0)
  );

  // head displacement: masked regions nudge toward cursor
  float headMask = texture(tMask, uvPixel).r;
  vec2 headOffset = uMouse * 0.006 * headMask;

  vec2 sampleUV = uvPixel + headOffset + circleColor.r * -0.05;
  vec4 colorNormal = texture(tMap, sampleUV);
  vec4 colorMirrored = texture(tMap, vec2(1.0 - sampleUV.x, sampleUV.y));

  // blend to mirrored version in masked regions when cursor is left of center
  // uMouse.x is ~-0.5 (far left) to ~+0.5 (far right), 0 = center
  // smoothstep gives a soft transition zone around center
  float mirrorBlend = smoothstep(0.05, -0.05, uMouse.x) * headMask;
  vec4 color = mix(colorNormal, colorMirrored, mirrorBlend);
  color.rgb = vec3(RGB2Gray(color.rgb));
  color.rgb = vec3(color.r + newNoise.r);

  fragColor = dither8x8(
    vec2(gl_FragCoord.x * 0.36, gl_FragCoord.y * 0.36),
    color
  );
  fragColor.a = 1.0;
}
`;

function compileShader(
  gl: WebGL2RenderingContext,
  type: number,
  source: string,
): WebGLShader {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Shader compile error: ${info}`);
  }
  return shader;
}

function createProgram(
  gl: WebGL2RenderingContext,
  vs: WebGLShader,
  fs: WebGLShader,
): WebGLProgram {
  const program = gl.createProgram()!;
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(`Program link error: ${info}`);
  }
  return program;
}

export function GrainCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext("webgl2", {
      alpha: true,
      antialias: false,
      premultipliedAlpha: false,
    });
    if (!gl) {
      console.warn("WebGL2 not available");
      return;
    }

    const vs = compileShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER);
    const fs = compileShader(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER);
    const program = createProgram(gl, vs, fs);

    const posLoc = gl.getAttribLocation(program, "position");
    const uvLoc = gl.getAttribLocation(program, "uv");

    const uScreen = gl.getUniformLocation(program, "uScreen");
    const uTextureSize = gl.getUniformLocation(program, "uTextureSize");
    const uTime = gl.getUniformLocation(program, "uTime");
    const uMouse = gl.getUniformLocation(program, "uMouse");
    const uZoom = gl.getUniformLocation(program, "uZoom");
    const uMap = gl.getUniformLocation(program, "tMap");
    const uMask = gl.getUniformLocation(program, "tMask");

    // fullscreen triangle (oversized to cover clip space)
    const verts = new Float32Array([
      -1, -1, 0, 0,
       3, -1, 2, 0,
      -1,  3, 0, 2,
    ]);

    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);

    const buf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);

    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 16, 0);
    gl.enableVertexAttribArray(uvLoc);
    gl.vertexAttribPointer(uvLoc, 2, gl.FLOAT, false, 16, 8);

    gl.bindVertexArray(null);

    const texture = gl.createTexture()!;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    // 1x1 placeholder
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));

    const texSize = { w: 1, h: 1 };
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = "/Clash/ClashHogMountain.png";
    img.onload = () => {
      texSize.w = img.naturalWidth;
      texSize.h = img.naturalHeight;
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    };

    const maskTexture = gl.createTexture()!;
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, maskTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));

    const maskImg = new Image();
    maskImg.crossOrigin = "anonymous";
    maskImg.src = "/Clash/white_mask.png";
    maskImg.onload = () => {
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, maskTexture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, maskImg);
    };

    const mouse = { x: 0, y: 0 };
    const targetMouse = { x: 0, y: 0 };
    let zoom = 0;
    let targetZoom = 0;
    let cancelled = false;

    const handlePointerMove = (e: PointerEvent) => {
      targetMouse.x = (e.clientX / window.innerWidth - 0.5);
      targetMouse.y = (e.clientY / window.innerHeight - 0.5) * -1;
    };
    const handleMouseEnter = () => { targetZoom = 1; };
    const handleMouseLeave = () => { targetZoom = 0; };

    document.body.addEventListener("pointermove", handlePointerMove, { passive: true });
    document.body.addEventListener("mouseenter", handleMouseEnter);
    document.body.addEventListener("mouseleave", handleMouseLeave);

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio, 2);
      const w = window.innerWidth;
      const h = window.innerHeight;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      gl.viewport(0, 0, canvas.width, canvas.height);
    };
    resize();
    window.addEventListener("resize", resize);

    // start with zoom at 0, animate to 1 after a beat
    targetZoom = 1;

    const render = (time: number) => {
      if (cancelled) return;

      mouse.x += (targetMouse.x - mouse.x) * 0.08;
      mouse.y += (targetMouse.y - mouse.y) * 0.08;
      zoom += (targetZoom - zoom) * 0.06;

      gl.useProgram(program);
      gl.uniform2f(uScreen, canvas.width, canvas.height);
      gl.uniform2f(uTextureSize, texSize.w, texSize.h);
      gl.uniform1f(uTime, time);
      gl.uniform2f(uMouse, mouse.x, mouse.y);
      gl.uniform1f(uZoom, zoom);
      gl.uniform1i(uMap, 0);
      gl.uniform1i(uMask, 1);

      gl.bindVertexArray(vao);
      gl.drawArrays(gl.TRIANGLES, 0, 3);

      requestAnimationFrame(render);
    };
    requestAnimationFrame(render);

    return () => {
      cancelled = true;
      document.body.removeEventListener("pointermove", handlePointerMove);
      document.body.removeEventListener("mouseenter", handleMouseEnter);
      document.body.removeEventListener("mouseleave", handleMouseLeave);
      window.removeEventListener("resize", resize);
      gl.deleteTexture(texture);
      gl.deleteTexture(maskTexture);
      gl.deleteBuffer(buf);
      gl.deleteVertexArray(vao);
      gl.deleteProgram(program);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 h-full w-full"
      style={{ display: "block" }}
    />
  );
}
