# MedImageFlow 开发日志

本文档仅用于本地开发管理，不得提交或公开发布。

## 2026-07-07：Image visualization

### 完成内容

- 新增 `visualization`：显示 2D 灰度或 RGB/RGBA 图像；对于 3D volume，显示
  axial、coronal 和 sagittal 三个中心截面。
- 新增 `mip_visualization`：沿 3D volume 的三个空间轴分别计算并显示最大强度投影。
- 支持 channels-last RGB/RGBA 2D 图像和 3D volume。
- 支持可选 spacing；提供 spacing 时按真实物理坐标与比例显示，否则使用图像 shape
  对应的 pixel/voxel 坐标。
- 支持通过 `figure_name` 自定义 Matplotlib figure 名称，通过 `show=False` 延迟显示。
- 将 Matplotlib 声明为 `visualization` 可选依赖，并纳入 `all` extra。
- 从 package 顶层导出 `visualization` 和 `mip_visualization`。
- 补充输入形状、中心截面、MIP、spacing、RGB volume、figure name 与异常输入测试。
- 更新英文与简体中文 README 的安装方式、接口说明和使用示例。

### 检查说明

- 按开发要求，本轮完成后未运行代码或测试，仅进行源码与文档检查。
