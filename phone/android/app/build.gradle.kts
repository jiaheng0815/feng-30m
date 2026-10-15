plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "com.feng.feng30m"
    compileSdk = 35
    // 与 CI / 文档里写的一致；装别的版本用 sdkmanager "ndk;<version>" 后改这里
    ndkVersion = "27.3.13750724"

    defaultConfig {
        applicationId = "com.feng.feng30m"
        minSdk = 26          // Android 8.0；NEON 是 arm64 基线
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"

        // 只出 arm64：引擎的 NEON 内核（ARMv8.2 dotprod）与权重格式都按 aarch64 设计，
        // 32 位 ABI 既没有 dotprod 也没必要（手机内存足够，模型 15MB 全量 mmap）。
        ndk {
            abiFilters += "arm64-v8a"
        }

        externalNativeBuild {
            cmake {
                // 引擎是纯 C23，不链 C++ 运行时：APK 里不会多出 libc++_shared.so
                arguments += listOf("-DANDROID_STL=none")
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false      // 没有反射/序列化，混淆收益不大，先关掉好排错
            isShrinkResources = false
        }
        debug {
            isJniDebuggable = true
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        compose = true
    }

    packaging {
        jniLibs.useLegacyPackaging = false
    }

    androidResources {
        // 模型走 assets，必须**不压缩**：Q4 权重本身接近高熵、压不动，
        // 压了只是白花构建时间；更重要的是不压缩才拿得到 openFd（精确长度 + 可映射）。
        noCompress += listOf("bin")
    }
}

dependencies {
    implementation(platform("androidx.compose:compose-bom:2024.10.01"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")

    debugImplementation("androidx.compose.ui:ui-tooling")
}
