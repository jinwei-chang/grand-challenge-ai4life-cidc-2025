import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)


    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)



class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv3D(in_channels, feature))
            in_channels = feature

        self.bottleneck = DoubleConv3D(features[-1], features[-1]*2)

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose3d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv3D(feature*2, feature))

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="trilinear", align_corners=True)
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)
    
    
class DoubleConvnmODE(nn.Module):
    """
    這就是標準的 3D 卷積塊，我們用它來當作 ODE 的微分函數 f(x, t)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        groups = 8 if out_channels % 8 == 0 else 1

        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            # nn.ReLU(inplace=True),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            # nn.ReLU(inplace=True)
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class ODEBlock(nn.Module):
    """
    【核心組件】: 定義 ODE 演化過程
    我們不依賴外部庫 (torchdiffeq)，直接手刻一個輕量的 RK4 Solver。
    這相當於一個「無限深」的 ResNet Block。
    """
    def __init__(self, channels):
        super().__init__()
        # 這裡定義微分函數 f(h, t)
        # 為了簡化，我們假設 f 不隨 t 變化 (Autonomous ODE)，或者把 t 視為隱含變數
        self.ode_func = DoubleConvnmODE(channels, channels)
        
        # 定義積分時間區間 (例如從 t=0 到 t=1)
        self.integration_time = 1.0 
        self.num_steps = 4  # 切成幾步積分 (步數越多越準，但越慢)

    def forward(self, x):
        # 初始狀態 h(0) = x
        h = x
        dt = self.integration_time / self.num_steps
        
        # RK4 積分迴圈 (Runge-Kutta 4th Order)
        for _ in range(self.num_steps):
            k1 = self.ode_func(h)
            k2 = self.ode_func(h + 0.5 * dt * k1)
            k3 = self.ode_func(h + 0.5 * dt * k2)
            k4 = self.ode_func(h + dt * k3)
            
            # 更新狀態
            h = h + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
        return h      


class UNetnmODE(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # --- Encoder (下採樣) ---
        for feature in features:
            self.downs.append(DoubleConvnmODE(in_channels, feature))
            in_channels = feature

        # --- Bottleneck (核心層) ---
        # 1. 先做一次卷積把特徵數翻倍
        self.bottleneck_conv = DoubleConvnmODE(features[-1], features[-1]*2)
        
        # 2. 插入 ODE Block (輸入輸出通道數一樣，進行特徵演化)
        self.bottleneck_ode = ODEBlock(features[-1]*2)

        # --- Decoder (上採樣) ---
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose3d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConvnmODE(feature*2, feature))

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # 1. Encoder Path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # 2. Bottleneck Path (With nmODE)
        x = self.bottleneck_conv(x) # 先變形
        x = self.bottleneck_ode(x)  # 再演化 (這是與普通 U-Net 的差異點)

        # 3. Decoder Path
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x) # Upsample
            skip_connection = skip_connections[idx//2]

            # 處理尺寸不匹配 (Padding/Interpolation)
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="trilinear", align_corners=True)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip) # DoubleConv

        return self.final_conv(x)
