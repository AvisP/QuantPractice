class Model3(nn.Module):

    def __init__(self, classify=False, embedding_size = 128, device=None):

        super(Model3, self).__init__()
        self.conv1 = nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.fc1 = nn.Sequential(nn.Linear(256*7*7, 2048), nn.ReLU(inplace=True), nn.Dropout())# maxout? #256 7 7
        self.fc7128 = nn.Sequential(nn.Linear(2048, embedding_size))

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.pool1(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)

        x = self.fc7128(x)
        x = self.dequant(x)

        x = nn.functional.normalize(x, p=2, dim=1)       
        return x


    def fuse_model(self):
        for m in self.modules():
            if type(m) == nn.Sequential:
                if type(m[0])==nn.BatchNorm2d:
                    # self.conv1[0] = nn.Identity()
                    torch.quantization.fuse_modules(self.conv1, ['1', '2', '3'], inplace=True)
                elif type(m[0])==nn.Conv2d and type(m[1])==nn.BatchNorm2d and type(m[2])==nn.ReLU:
                    torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
                elif (type(m[0])==nn.Linear and len(m)>1):
                    torch.quantization.fuse_modules(m, ['0', '1'], inplace=True)
                else:       
                    print ('No fusion performed on this layer')
                    print(m)
        print('Fusion Complete')