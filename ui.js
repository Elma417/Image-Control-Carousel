/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as tf from '@tensorflow/tfjs';

const CONTROLS = ['up', 'down', 'left', 'right'];
//这是具体的PACMAN操作码 没有使用的必要。
// const CONTROL_CODES = [38, 40, 37, 39];
//初始化
export function init() {
  document.getElementById('controller').style.display = '';
  statusElement.style.display = 'none';
}

const trainStatusElement = document.getElementById('train-status');

//从UI设置超参数
const learningRateElement = document.getElementById('learningRate');
export const getLearningRate = () => +learningRateElement.value;

const batchSizeFractionElement = document.getElementById('batchSizeFraction');
export const getBatchSizeFraction = () => +batchSizeFractionElement.value;

const epochsElement = document.getElementById('epochs');
export const getEpochs = () => +epochsElement.value;

const denseUnitsElement = document.getElementById('dense-units');
export const getDenseUnits = () => +denseUnitsElement.value;
const statusElement = document.getElementById('status');

//开始Pacman游戏
// export function startPacman() {
//   google.pacman.startGameplay();
// }

//节流
// export function throttle(fn, gap) {
//   let timerId = null;
//   return function (...rest) {
//     if(timerId === null) {
//       fn(...rest) // 立即执行
//       timerId = setTimeout(() => {
//         // 在间隔时间后清除标识
//         timerId = null;
//       }, gap)
//     }
//   }
// }

//控制轮播图
export function predictClass(classId) {
  switch (classId)
  {
    case 0 :
      $("#myCarousel").carousel('cycle');
      break;
    case 1:
      $("#myCarousel").carousel('pause');
      break;
    case 2:
      $("#myCarousel").carousel('prev');
      break;
    case 3:
      $("#myCarousel").carousel('next');
      break;
    default:
      break;
  }
  console.log(classId);
  document.body.setAttribute('data-active', CONTROLS[classId]);
}

//正在预测 应该是那个高亮？
export function isPredicting() {
  statusElement.style.visibility = 'visible';
}

//完成预测
export function donePredicting() {
  statusElement.style.visibility = 'hidden';
}

//显示训练状态
export function trainStatus(status) {
  trainStatusElement.innerText = status;
}

//初始化获取训练集的按钮
export let addExampleHandler;
export function setExampleHandler(handler) {
  addExampleHandler = handler;
}
let mouseDown = false;
const totals = [0, 0, 0, 0];

const upButton = document.getElementById('up');
const downButton = document.getElementById('down');
const leftButton = document.getElementById('left');
const rightButton = document.getElementById('right');

const thumbDisplayed = {};


//handler处理图像 真正捕获训练集
async function handler(label) {
  mouseDown = true;
  const className = CONTROLS[label];
  const button = document.getElementById(className);
  const total = document.getElementById(className + '-total');
  while (mouseDown) {
    addExampleHandler(label);
    document.body.setAttribute('data-active', CONTROLS[label]);
    total.innerText = ++totals[label];
    await tf.nextFrame();
  }
  document.body.removeAttribute('data-active');
}

//绑定按钮的鼠标点击事件
upButton.addEventListener('mousedown', () => handler(0));
upButton.addEventListener('mouseup', () => mouseDown = false);

downButton.addEventListener('mousedown', () => handler(1));
downButton.addEventListener('mouseup', () => mouseDown = false);

leftButton.addEventListener('mousedown', () => handler(2));
leftButton.addEventListener('mouseup', () => mouseDown = false);

rightButton.addEventListener('mousedown', () => handler(3));
rightButton.addEventListener('mouseup', () => mouseDown = false);

//绘制拍照的缩略图
export function drawThumb(img, label) {
  if (thumbDisplayed[label] == null) {
    const thumbCanvas = document.getElementById(CONTROLS[label] + '-thumb');
    draw(img, thumbCanvas);
  }
}

//使用canvas绘制图像 将图像大小格式化为缩略图的大小
export function draw(image, canvas) {
  const [width, height] = [224, 224];
  const ctx = canvas.getContext('2d');
  const imageData = new ImageData(width, height);
  const data = image.dataSync();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    imageData.data[j + 0] = (data[i * 3 + 0] + 1) * 127;
    imageData.data[j + 1] = (data[i * 3 + 1] + 1) * 127;
    imageData.data[j + 2] = (data[i * 3 + 2] + 1) * 127;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}
