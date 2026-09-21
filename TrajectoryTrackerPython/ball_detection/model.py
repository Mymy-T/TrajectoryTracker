import cv2
import numpy as np
import torch
import torch.nn as nn
import settings

# Credit to yastrebksv's implementation of TrackNet on github: https://github.com/yastrebksv/TrackNet/blob/main/model.py

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pad=1, stride=1, bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.block(x)

class TrackNet(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        
        self.out_channels = out_channels

        self.conv1 = ConvBlock(in_channels=9, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(in_channels=64, out_channels=128)
        self.conv4 = ConvBlock(in_channels=128, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = ConvBlock(in_channels=128, out_channels=256)
        self.conv6 = ConvBlock(in_channels=256, out_channels=256)
        self.conv7 = ConvBlock(in_channels=256, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv8 = ConvBlock(in_channels=256, out_channels=512)
        self.conv9 = ConvBlock(in_channels=512, out_channels=512)
        self.conv10 = ConvBlock(in_channels=512, out_channels=512)
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(in_channels=512, out_channels=256)
        self.conv12 = ConvBlock(in_channels=256, out_channels=256)
        self.conv13 = ConvBlock(in_channels=256, out_channels=256)
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(in_channels=256, out_channels=128)
        self.conv15 = ConvBlock(in_channels=128, out_channels=128)
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(in_channels=128, out_channels=64)
        self.conv17 = ConvBlock(in_channels=64, out_channels=64)
        self.conv18 = ConvBlock(in_channels=64, out_channels=self.out_channels)

        self.softmax = nn.Softmax(dim=1) # Channel dimension
        self._init_weights()
                  
    def forward(self, x, testing=False): 
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)    
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.ups1(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.ups2(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.ups3(x)
        x = self.conv16(x)
        x = self.conv17(x)
        out = self.conv18(x)
        # out = x.reshape(batch_size, self.out_channels, -1)
        if testing:
            out = self.softmax(out)
        return out # Shape is (batch_size, out_channels, h, w); for this project, it's (batch_size, 256, 360, 640)
    
    def process_output(self, probs):
        """
        Shape is transformed from (num_batches, out_channels, h, w) to (num_batches, h, w) where the probs along the channel dimension are collapsed using argmax
        """
        out = probs.detach().cpu().numpy()
        out = np.argmax(out, axis=1)
        return out
    
    def filter_by_threshold(self, output, threshold=100):
        # For model output, map any pixels with value < t to 0, otherwise 255 (after using softmax) where t is threshold; in paper, t=128; adjusted experimentally
        return np.where(output >= threshold, 255, 0).astype(np.uint8) # uint8 required for cv2.HoughCircles

    def obtain_ball_center_coords(self, filtered_output):
        """
        Returns a list of the detected ball's center x, center y, and radius; returns None if no ball or more than 1 ball is detected
        """
        # Use Hough Gradient Method, and return the center coords of the ball; make sure only one circle is detected, otherwise no ball
        circles = cv2.HoughCircles( # Shape of circles is (1, n_circles_detected, 3); the first dim is byproduct of underlying implementation; the third consists of [x, y, radius]
            filtered_output,          # Your normalized, single-channel TrackNet heatmap
            method=cv2.HOUGH_GRADIENT,
            dp=1,                  # Keep at 1 for exact pixel accuracy
            minDist=15,            # Only 1 ball should be found
            param1=50,             # Sharpness of transitions, higher means sharper edges detected
            param2=4,             # Low value takens into account small/faint ball blobs when ball is far away
            minRadius=2,           # Accounts for the ball when it is furthest from camera
            maxRadius=30           # Accounts for the ball when it is closest to camera
        )

        if circles is None:
            return None
        
        if circles.shape[1] != 1:
            return None
        
        return list(circles[0][0]) # Returns [x, y, radius]
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(module.weight, -0.05, 0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    


    
if __name__ == '__main__':
    device =settings.device
    print(f"Testing on device: {device.type}")
    model = TrackNet().to(device)
    input = torch.rand(1, 9, 360, 640).to(device)
    out = model(input) # Shape should be (1, 256, 360, 640)
    testing_out = np.squeeze(model.process_output(out), axis=0) # Shape should be (360, 640)
    print(f"out shape = {out.shape}")
    print(f"testing mode out shape = {testing_out.shape}")
    
    