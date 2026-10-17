这个目录放的是模型文件（model.bin / tokenizer.bin），它们不入 git。

构建 APK 时由 phone/tools/build_apk.ps1 自动从模型导出目录拷进来：

    .\phone\tools\build_apk.ps1                      # 默认用 board/model_export_v3_19b6
    .\phone\tools\build_apk.ps1 -ModelDir <导出目录>  # 或自己指定

也可以用 Android Studio 构建，但同样要先把两个 .bin 放到这里
（从 Release 的 feng-30m-c-engine-model-v3.19-embed.zip 里解出来即可）。

为什么要放 assets 而不是运行时下载：这个 App 完全离线、不申请网络权限。
首次启动时把 assets 解包到 filesDir，之后用 mmap 只读映射，不占 Java 堆。
