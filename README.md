### 项目描述
通过使用tensorflow.js 建立相应的模型，通过建立的模型识别出来对应的4种不同的动作，从而触发相应的照片墙的行为。  
先在摄像头开启的情况下，点击下方右侧的四个按钮收集样本，它们分别对应4种不同的操作，收集到足够的样本后开始训练模型，模型训练完毕后，通过摄像头识别用户的动作，当用户做出相应的动作后，页面上的照片墙能够做出相应的操作。  
可以通过设置的向左、向右的手势来控制照片墙左右变化，通过设置向上的手势开始自动轮播，向下的手势停止自动轮播。


### 项目部署
webIDE不能运行，本地安装依赖后打开index.html即可。
```javascript
// 安装依赖
yarn install
yarn watch
``` 
（尝试往Vue3迁移，但好多问题解决不了时间不允许了T-T）