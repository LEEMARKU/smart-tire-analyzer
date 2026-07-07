package com.example.util

import android.app.ActivityManager
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.os.Environment
import android.os.StatFs
import android.os.storage.StorageManager

data class DeviceSpecs(
    val totalRamMb: Long,
    val availableRamMb: Long,
    val totalStorageMb: Long,
    val availableStorageMb: Long,
    val androidSdk: Int,
    val hasGpu: Boolean,
    val hasNnapi: Boolean,
    val numCores: Int,
    val isLowRam: Boolean,
    val has64BitAbi: Boolean,
)

object SystemSpecProvider {

    fun getDeviceSpecs(context: Context): DeviceSpecs {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)

        val totalRamMb = memInfo.totalMem / (1024 * 1024)
        val availableRamMb = memInfo.availMem / (1024 * 1024)

        val storageStats = getStorageStats(context)
        val numCores = Runtime.getRuntime().availableProcessors()

        return DeviceSpecs(
            totalRamMb = totalRamMb,
            availableRamMb = availableRamMb,
            totalStorageMb = storageStats.first,
            availableStorageMb = storageStats.second,
            androidSdk = Build.VERSION.SDK_INT,
            hasGpu = hasGpu(context),
            hasNnapi = Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1,
            numCores = numCores,
            isLowRam = activityManager.isLowRamDevice,
            has64BitAbi = has64Bit(),
        )
    }

    private fun getStorageStats(context: Context): Pair<Long, Long> {
        return try {
            val storageManager = context.getSystemService(Context.STORAGE_SERVICE) as StorageManager
            val storageVolume = storageManager.storageVolumes.firstOrNull()
            if (storageVolume != null && Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                val stat = StatFs(storageVolume.directory?.absolutePath ?: Environment.getDataDirectory().absolutePath)
                val blockSize = stat.blockSizeLong
                val totalBlocks = stat.blockCountLong
                val availableBlocks = stat.availableBlocksLong
                Pair(
                    (totalBlocks * blockSize) / (1024 * 1024),
                    (availableBlocks * blockSize) / (1024 * 1024)
                )
            } else {
                val stat = StatFs(Environment.getDataDirectory().absolutePath)
                val blockSize = stat.blockSizeLong
                val totalBlocks = stat.blockCountLong
                val availableBlocks = stat.availableBlocksLong
                Pair(
                    (totalBlocks * blockSize) / (1024 * 1024),
                    (availableBlocks * blockSize) / (1024 * 1024)
                )
            }
        } catch (e: Exception) {
            Pair(0L, 0L)
        }
    }

    private fun hasGpu(context: Context): Boolean {
        return try {
            val pm = context.packageManager
            pm.hasSystemFeature("android.hardware.opengles.aep") ||
                Build.VERSION.SDK_INT >= Build.VERSION_CODES.O &&
                pm.hasSystemFeature("android.hardware.vulkan.version")
        } catch (e: Exception) {
            false
        }
    }

    private fun has64Bit(): Boolean {
        return try {
            val abis = Build.SUPPORTED_64_BIT_ABIS
            abis.isNotEmpty()
        } catch (e: Exception) {
            false
        }
    }

    fun getDeviceTier(specs: DeviceSpecs): String {
        return when {
            specs.totalRamMb >= 6144 && specs.androidSdk >= 28 && specs.hasGpu -> "high"
            specs.totalRamMb >= 4096 && specs.androidSdk >= 28 -> "high_medium"
            specs.totalRamMb >= 3072 && specs.androidSdk >= 26 -> "medium"
            specs.totalRamMb >= 2048 && specs.androidSdk >= 24 -> "low_medium"
            else -> "low"
        }
    }

    fun isGpuAccelerationSupported(specs: DeviceSpecs): Boolean {
        return specs.hasGpu && specs.androidSdk >= Build.VERSION_CODES.O_MR1
    }

    fun isNnapiSupported(specs: DeviceSpecs): Boolean {
        return specs.hasNnapi && specs.androidSdk >= Build.VERSION_CODES.O_MR1
    }
}
