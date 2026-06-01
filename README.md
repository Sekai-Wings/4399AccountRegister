# 4399AccountRegister

> 重度Vibe Coding项目，嫌弃勿骂，玻璃心

> 基于 **Golang** 的纯协议全自动4399账号注册机

# 使用方法

由于该项目使用了 **onnxruntime_go** 进行图形验证码识别，所以你需要先从 [ONNX Runtime仓库](https://github.com/microsoft/onnxruntime) 下载最新最热的ONNX Runtime，然后提取出onnxruntime so/dll，放入该项目文件夹，然后在config.json中填入相对路径，修改想要的参数，随后下载一下库run即可。

如果你使用的是 **Termux** ，不妨看看 [ONNXRuntime-Termux](https://github.com/Sekai-Wings/onnxruntime-termux)。

# Tips

**大神4399最近注册风控严重，本项目暂未支持IP池代理(懒得prompt)，有志之士可以自己修改原始密码以支持代理IP池。**

**4399ocr文件夹内验证码模型来自 [4399Register](https://github.com/boluoreg/4399Register) 该部分保留原协议 [SKY License](https://github.com/boluoreg/4399Register/blob/main/LICENSE) 如有侵权请联系删除**

# 致谢

[4399Register](https://github.com/boluoreg/4399Register)：使用了该项目自训练的4399OCR模型，且参考了部分代码，感激不尽
