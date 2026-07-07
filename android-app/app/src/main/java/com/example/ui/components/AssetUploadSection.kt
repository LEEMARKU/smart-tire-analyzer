package com.example.ui.components

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AddAPhoto
import androidx.compose.material.icons.filled.CloudUpload
import androidx.compose.material.icons.filled.Image
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.example.viewmodel.TireTwinViewModel

@Composable
fun AssetUploadSection(
    viewModel: TireTwinViewModel,
    modifier: Modifier = Modifier
) {
    val frontImageUri by viewModel.frontImageUri.collectAsState()
    val sidewallImageUri by viewModel.sidewallImageUri.collectAsState()

    val frontImageLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        viewModel.setFrontImageUri(uri)
    }

    val sidewallImageLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        viewModel.setSidewallImageUri(uri)
    }

    Card(
        modifier = modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(alpha = 0.2f))
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Icon(
                    imageVector = Icons.Default.CloudUpload,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.size(20.dp)
                )
                Text(
                    text = "TIRE VISUAL EVIDENCE ASSETS",
                    style = MaterialTheme.typography.labelSmall.copy(
                        fontFamily = FontFamily.Monospace,
                        fontWeight = FontWeight.Bold,
                        letterSpacing = 1.5.sp,
                        color = MaterialTheme.colorScheme.primary
                    )
                )
            }

            Text(
                text = "Upload reference images for multi-angle AI analysis and historical documentation.",
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                fontSize = 10.sp,
                fontFamily = FontFamily.Monospace,
                lineHeight = 14.sp
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                ImageUploadBox(
                    label = "FRONT IMAGE",
                    imageUri = frontImageUri,
                    onClick = { frontImageLauncher.launch("image/*") },
                    modifier = Modifier.weight(1f)
                )
                ImageUploadBox(
                    label = "SIDEWALL IMAGE",
                    imageUri = sidewallImageUri,
                    onClick = { sidewallImageLauncher.launch("image/*") },
                    modifier = Modifier.weight(1f)
                )
            }
        }
    }
}

@Composable
fun ImageUploadBox(
    label: String,
    imageUri: Uri?,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier,
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(120.dp)
                .clip(RoundedCornerShape(8.dp))
                .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f))
                .border(
                    width = 1.dp,
                    color = if (imageUri != null) MaterialTheme.colorScheme.primary.copy(alpha = 0.5f) 
                            else MaterialTheme.colorScheme.outline.copy(alpha = 0.2f),
                    shape = RoundedCornerShape(8.dp)
                )
                .clickable { onClick() },
            contentAlignment = Alignment.Center
        ) {
            if (imageUri != null) {
                AsyncImage(
                    model = imageUri,
                    contentDescription = label,
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Crop
                )
                // Overlay for re-upload
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(Color.Black.copy(alpha = 0.3f)),
                    contentAlignment = Alignment.BottomCenter
                ) {
                    Text(
                        text = "CHANGE",
                        color = Color.White,
                        fontSize = 9.sp,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.padding(bottom = 4.dp)
                    )
                }
            } else {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.AddAPhoto,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f),
                        modifier = Modifier.size(24.dp)
                    )
                    Text(
                        text = "SELECT",
                        fontSize = 10.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
        }
        Text(
            text = label,
            fontSize = 9.sp,
            fontFamily = FontFamily.Monospace,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center
        )
    }
}
