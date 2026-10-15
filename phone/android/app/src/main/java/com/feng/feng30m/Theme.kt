package com.feng.feng30m

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext

private val Light = lightColorScheme(
    primary = Color(0xFF2E6B4F),
    primaryContainer = Color(0xFFCDEBD9),
    surfaceVariant = Color(0xFFE7E9E4),
)

private val Dark = darkColorScheme(
    primary = Color(0xFF8FD3AE),
    primaryContainer = Color(0xFF1F5238),
    surfaceVariant = Color(0xFF2A2E2B),
)

@Composable
fun FengTheme(content: @Composable () -> Unit) {
    val dark = isSystemInDarkTheme()
    val ctx = LocalContext.current
    val scheme = when {
        // Android 12+ 跟随系统取色，低版本用上面那套固定的
        Build.VERSION.SDK_INT >= Build.VERSION_CODES.S ->
            if (dark) dynamicDarkColorScheme(ctx) else dynamicLightColorScheme(ctx)
        dark -> Dark
        else -> Light
    }
    MaterialTheme(colorScheme = scheme, content = content)
}
