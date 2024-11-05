# MNIST

Bu proje, basit bir sinir ağı  kullanarak MNIST veri seti üzerinde el yazısı rakamları tanımak için eğitilen bir model içermektedir. Projede PyTorch kullanılarak bir model eğitilmiş ve test edilmiştir. Aşağıda projenin dosya yapısı, kullanımı ve örnek tahminler hakkında bilgiler verilmiştir.

## Dosya Yapısı

- **0.png, 4.png**: Modelin test edilmesi için kullanılan örnek el yazısı rakam görüntüleri.
- **mnist.py**: MNIST veri seti üzerinde basit bir sinir ağı ile model eğitimi yapan Python kodu.
- **simple_net_mnist.pth**: Eğitilmiş modelin ağırlıklarını içeren dosya.
- **test_model.py**: Eğitilmiş modeli yükleyip test etmek için örnek görüntüler üzerinde tahmin yapan kod.


## Kullanım

1. Bu projeyi klonlayın:
   ```bash
   git clone https://github.com/hqkqn32/mnist.git
   pip install torch torchvision matplotlib
   Modeli Eğitme
   Eğer modeli baştan eğitmek isterseniz mnist.py dosyasını çalıştırabilirsiniz.
   Eğitilmiş modeli kullanarak tahminler yapmak için test_model.py dosyasını çalıştırın:

