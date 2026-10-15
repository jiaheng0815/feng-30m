// 顶层构建脚本：只声明插件版本，模块配置见 app/build.gradle.kts。
//
// 版本组合是保守可用的那一组：
//   AGP 8.7.3 + Kotlin 2.0.21 + Compose BOM 2024.10.01 + Gradle 8.9 + compileSdk 35
// JDK 要求 17~21（AGP 8.x 不支持 JDK 24+；Android Studio 自带的 JBR 21 正好）。
plugins {
    id("com.android.application") version "8.7.3" apply false
    id("org.jetbrains.kotlin.android") version "2.0.21" apply false
    id("org.jetbrains.kotlin.plugin.compose") version "2.0.21" apply false
}
