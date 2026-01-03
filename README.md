 Deep Learning Based Multi-Modal Steganography

 Abstract
In the digital era, data security and confidentiality have become critical due to the exponential increase in information exchange over open networks. Steganography conceals secret information within digital media, ensuring secure communication without revealing the existence of the data.
This project presents a **Deep Learning-Based Multi-Modal Steganography system** that hides and retrieves multiple types of secret data—including **text, image, audio, and video**—within a single cover image or video. The system integrates **Convolutional Neural Networks (CNNs)** and **autoencoder-based architectures** to preserve perceptual quality while ensuring robustness and security. A password-based encryption layer enhances protection against unauthorized access.

 Objectives
- Enable secure communication through steganography
- Support **multi-modal data hiding** (text, image, audio, video)
- Preserve visual, audio, and temporal quality of cover media
- Integrate deep learning–based steganalysis for validation
- Provide a scalable and robust framework for real-world use

 Technologies Used
- **Python**
- **PyTorch**
- **Deep Learning (CNN, Autoencoders, LSTM)**
- **Flask & Socket.IO**
- **MongoDB**
- **Git & Git LFS**

 System Architecture
The system uses modality-specific deep encoders and decoders:
- **Images**: CNN-based embedding and extraction
- **Audio**: Spectrogram-based CNN processing
- **Video**: Hybrid 3D-CNN + LSTM architecture
- **Text**: Character-level reversible transformation
A shared latent space enables efficient multi-modal embedding with minimal distortion.

 Methodology
1. Secret data is compressed using Huffman coding.
2. Cover media is preprocessed (normalization, frame extraction, spectrogram conversion).
3. Deep encoders extract high-level features.
4. Secret data is embedded into feature representations.
5. Deep learning steganalysis validates imperceptibility.
6. Receiver extracts the secret using the correct password.

A multi-objective loss function ensures:
- Low distortion
- High payload recovery
- Robustness against attacks

 Project Structure
├── AES.py # Encryption module
├── app.py # Flask application
├── image_embed.py # Image embedding
├── image_extract.py # Image extraction
├── audio_embed.py # Audio embedding
├── audio_extract.py # Audio extraction
├── video_embed.py # Video embedding
├── video_extract.py # Video extraction
├── sender.py # Sender module
├── receiver.py # Receiver module
├── train_all_models.py # Training script
├── stego_detector.pth # Image steganalysis model
├── stego_detector_r3d18.pth # Video detection model
├── stego_multi_modal.pth # Multi-modal model
├── secret_size.txt # Payload size info
└── .gitattributes # Git LFS configuration

 How to Run
git clone https://github.com/supraja-shetty/Deep-learning-based-multi-modal-steganography.git
cd Deep-learning-based-multi-modal-steganography
python app.py

Results & Discussion
Image Steganography
PSNR: 52.7 dB
SSIM: 0.987
Detection accuracy: 98.1%

Video Steganography
Average PSNR: 46.2 dB
Detection accuracy: 90.5%

Audio Steganography
Detection accuracy: 97.3%
Imperceptible waveform distortion

Text Steganography
100% extraction accuracy with correct password

The system demonstrates high imperceptibility, robustness, and reliable payload recovery.
Conclusion

The proposed Deep Learning-based Multi-Modal Steganography system effectively integrates neural networks with multimedia processing to achieve secure and imperceptible data hiding. By leveraging CNNs and LSTM architectures, the system supports multiple media types while maintaining high fidelity and resistance to attacks.

