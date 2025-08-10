import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torchvision.models as models

class EnhancedASPP(nn.Module):
    """
    Enhanced Atrous Spatial Pyramid Pooling module with improved feature extraction.
    """
    
    def __init__(self, in_channels, out_channels, atrous_rates=[6, 12, 18]):
        super(EnhancedASPP, self).__init__()
        
        # 1x1 convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, track_running_stats=False)
        
        # Atrous convolutions with different rates
        self.atrous_convs = nn.ModuleList()
        for rate in atrous_rates:
            self.atrous_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels, track_running_stats=False),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Global average pooling branch - use adaptive pooling to avoid 1x1 issues
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),  # Use 4x4 instead of 1x1 to avoid batch norm issues
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        
        # Final 1x1 convolution
        self.conv_final = nn.Conv2d(out_channels * (len(atrous_rates) + 2), out_channels, 1, bias=False)
        self.bn_final = nn.BatchNorm2d(out_channels, track_running_stats=False)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        size = x.size()[2:]
        
        # 1x1 convolution
        conv1 = self.conv1(x)
        conv1 = self.bn1(conv1)
        conv1 = F.relu(conv1)
        
        # Atrous convolutions
        atrous_features = []
        for atrous_conv in self.atrous_convs:
            atrous_features.append(atrous_conv(x))
        
        # Global average pooling - handle small spatial dimensions
        if size[0] <= 4 or size[1] <= 4:
            # For very small feature maps, use a different approach
            global_feat = F.adaptive_avg_pool2d(x, (4, 4))
            global_feat = self.global_avg_pool(global_feat)
        else:
            global_feat = self.global_avg_pool(x)
        
        # Upsample global features to match input size
        global_feat = F.interpolate(global_feat, size=size, mode='bilinear', align_corners=False)
        
        # Concatenate all features
        concat_features = torch.cat([conv1] + atrous_features + [global_feat], dim=1)
        
        # Final convolution
        output = self.conv_final(concat_features)
        output = self.bn_final(output)
        output = self.dropout(output)
        
        return output


class AttentionModule(nn.Module):
    """
    Attention module to focus on important features and spatial regions.
    """
    
    def __init__(self, in_channels, reduction=16):
        super(AttentionModule, self).__init__()
        
        # Use minimum pooling size to avoid 1x1 issues
        self.avg_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.max_pool = nn.AdaptiveMaxPool2d((4, 4))
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Handle small spatial dimensions
        size = x.size()[2:]
        if size[0] <= 4 or size[1] <= 4:
            # For very small feature maps, use adaptive pooling
            avg_out = self.fc(self.avg_pool(x))
            max_out = self.fc(self.max_pool(x))
        else:
            avg_out = self.fc(self.avg_pool(x))
            max_out = self.fc(self.max_pool(x))
        
        attention = self.sigmoid(avg_out + max_out)
        
        # Upsample attention to match input size
        attention = F.interpolate(attention, size=size, mode='bilinear', align_corners=False)
        
        return x * attention


class EnhancedDecoder(nn.Module):
    """
    Enhanced decoder with attention mechanisms and skip connections.
    """
    
    def __init__(self, low_level_channels, aspp_channels, decoder_channels=256):
        super(EnhancedDecoder, self).__init__()
        
        self.conv_low_level = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        
        self.conv_aspp = nn.Sequential(
            nn.Conv2d(aspp_channels, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        
        self.attention = AttentionModule(decoder_channels)
        
        self.conv_final = nn.Sequential(
            nn.Conv2d(decoder_channels + 48, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, aspp_features, low_level_features):
        # Process ASPP features
        aspp_features = self.conv_aspp(aspp_features)
        aspp_features = self.attention(aspp_features)
        
        # Upsample ASPP features
        aspp_features = F.interpolate(aspp_features, size=low_level_features.size()[2:], 
                                    mode='bilinear', align_corners=False)
        
        # Process low-level features
        low_level_features = self.conv_low_level(low_level_features)
        
        # Concatenate and process
        concat_features = torch.cat([aspp_features, low_level_features], dim=1)
        output = self.conv_final(concat_features)
        
        return output


class EnhancedDeepLabV3Plus(nn.Module):
    """
    Enhanced DeepLabV3+ with improved ASPP, decoder, and attention mechanisms.
    """
    
    def __init__(self, num_classes=4, pretrained=True):
        super(EnhancedDeepLabV3Plus, self).__init__()
        
        # Load pretrained ResNet50 backbone
        if pretrained:
            self.backbone = models.resnet50(pretrained=True)
        else:
            self.backbone = models.resnet50(pretrained=False)
        
        # Extract layers for skip connections
        self.layer0 = nn.Sequential(self.backbone.conv1, self.backbone.bn1, 
                                   self.backbone.relu, self.backbone.maxpool)
        self.layer1 = self.backbone.layer1  # 256 channels
        self.layer2 = self.backbone.layer2  # 512 channels
        self.layer3 = self.backbone.layer3  # 1024 channels
        self.layer4 = self.backbone.layer4  # 2048 channels
        
        # Enhanced ASPP
        self.aspp = EnhancedASPP(2048, 256)
        
        # Enhanced Decoder
        self.decoder = EnhancedDecoder(256, 256, 256)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize the weights of the model."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        input_shape = x.shape[-2:]
        
        # Encoder
        x0 = self.layer0(x)      # 64 channels
        x1 = self.layer1(x0)     # 256 channels
        x2 = self.layer2(x1)     # 512 channels
        x3 = self.layer3(x2)     # 1024 channels
        x4 = self.layer4(x3)     # 2048 channels
        
        # ASPP
        aspp_features = self.aspp(x4)
        
        # Decoder with skip connection
        decoder_features = self.decoder(aspp_features, x1)
        
        # Final classification
        output = self.classifier(decoder_features)
        
        # Upsample to input size
        output = F.interpolate(output, size=input_shape, mode='bilinear', align_corners=False)
        
        return {'out': output}


class LightweightDeepLabV3Plus(nn.Module):
    """
    Lightweight version of DeepLabV3+ for faster inference.
    """
    
    def __init__(self, num_classes=4, pretrained=True):
        super(LightweightDeepLabV3Plus, self).__init__()
        
        # Use MobileNetV2 as backbone for lightweight version
        if pretrained:
            self.backbone = models.mobilenet_v2(pretrained=True)
        else:
            self.backbone = models.mobilenet_v2(pretrained=False)
        
        # Extract features
        self.features = self.backbone.features
        
        # Lightweight ASPP - use safer pooling
        self.aspp = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),  # Use 4x4 instead of 1x1
            nn.Conv2d(1280, 256, 1, bias=False),
            nn.BatchNorm2d(256, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Lightweight decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        
        # Extract features
        features = self.features(x)
        
        # ASPP - handle small spatial dimensions
        if features.size()[2] <= 4 or features.size()[3] <= 4:
            # For very small feature maps, use adaptive pooling
            aspp_features = F.adaptive_avg_pool2d(features, (4, 4))
            aspp_features = self.aspp(aspp_features)
        else:
            aspp_features = self.aspp(features)
        
        # Upsample
        aspp_features = F.interpolate(aspp_features, size=input_shape, 
                                    mode='bilinear', align_corners=False)
        
        # Decoder
        output = self.decoder(aspp_features)
        
        return {'out': output}


def get_enhanced_model(model_type='enhanced', num_classes=4, pretrained=True):
    """
    Factory function to get the specified enhanced model.
    
    Args:
        model_type: Type of model ('enhanced', 'lightweight', 'standard')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
    
    Returns:
        Model instance
    """
    
    if model_type == 'enhanced':
        return EnhancedDeepLabV3Plus(num_classes=num_classes, pretrained=pretrained)
    elif model_type == 'lightweight':
        return LightweightDeepLabV3Plus(num_classes=num_classes, pretrained=pretrained)
    elif model_type == 'standard':
        model = deeplabv3_resnet50(pretrained=pretrained)
        model.classifier = DeepLabHead(2048, num_classes)
        return model
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model):
    """Get detailed information about the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
    }
