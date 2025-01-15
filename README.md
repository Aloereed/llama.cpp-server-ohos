
# Llama.cpp Server for HarmonyOS Next

🌟 **Llama.cpp Server** 是一个高性能、轻量级的推理引擎，支持基于 Llama 模型的本地推理。此项目是 Llama.cpp Server 在 HarmonyOS Next 平台上的实现，充分利用鸿蒙分布式能力和硬件加速特性，为开发者提供高效的 AI 推理服务。

---

## 功能特性

- 🚀 **高性能推理**：基于 Llama.cpp 推理引擎，优化适配鸿蒙设备的多核和硬件加速能力。
- 🌐 **轻量级服务**：实现低功耗运行，适合边缘计算场景。


---

## 环境要求

在开始使用之前，请确保你的环境满足以下要求：

- **开发环境**
  - HarmonyOS DevEco Studio
  - OpenHarmony SDK
  - C++17 编译器（支持 LLVM 或 GCC）
- **硬件支持**
  - HarmonyOS 设备（例如：鸿蒙手机、平板、IoT 设备等）
- **依赖库**
  - Llama.cpp (已包含在本项目中)
  - OpenMP (用于多线程加速)

---

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/Aloereed/llama.cpp-server-ohos.git
cd llama.cpp-server-ohos
```

### 2. 编译项目

确保已安装 HarmonyOS 编译工具链和 CMake。



### 3. 部署到鸿蒙设备

通过 DevEco Studio 或命令行将编译生成的二进制文件部署到鸿蒙设备。


### 4. 调用服务

服务启动后，默认监听 `127.0.0.1:8000`。你可以通过Chatbox、OpenWebUI等使用 HTTP 调用推理接口。


---

## 使用指南

### 配置文件

在项目根目录下缺少build-profile，你需要新建一个空项目并复制它到这里。



## 开发计划

- [x] 实现 Llama.cpp 的基础推理功能
- [ ] 提供 RESTful 接口
- [ ] 支持分布式推理
- [ ] 增加多模型并发支持
- [ ] 提供图形化管理界面

---

## 贡献指南

欢迎任何形式的贡献！如果你有想法或发现问题，请提交 [Issue](https://github.com/Aloereed/llama.cpp-server-ohos/issues) 或发起 Pull Request。

---

## 许可证

本项目基于 [MIT License](LICENSE) 开源。

---

## 联系我们

如果你有任何问题或建议，请通过以下方式联系我们：

- 项目主页: [GitHub](https://github.com/Aloereed/llama.cpp-server-ohos)

---

希望你喜欢在 HarmonyOS 上体验 Llama.cpp Server 的高效推理！✨

