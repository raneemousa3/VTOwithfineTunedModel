# Virtual Fitting Room (VTO)

An AI-powered virtual fitting room application that uses computer vision to extract body measurements and provide accurate size recommendations for online shopping.

## Features

- Real-time body measurement extraction using computer vision
- Support for both image and video input
- Accurate pose detection and body segmentation
- Gender-specific measurement adjustments
- Confidence scoring for measurement accuracy
- Unit conversion (cm/inches)
- Calibration using reference objects

## Tech Stack

- FastAPI for the backend API
- MediaPipe for pose estimation and body segmentation
- OpenCV for image processing
- TensorFlow for measurement extraction models
- Python 3.8+

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/virtual-fitting-room.git
cd virtual-fitting-room
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Run the application:
```bash
python app.py
```

## Usage

The API provides endpoints for:
- Processing single images
- Processing video streams
- Getting measurement results
- Converting between measurement units

See the API documentation at `/docs` when running the server.

## Project Structure

```
virtual_fitting_room/
├── app/
│   ├── api/        # API routes and endpoints
│   ├── core/       # Core business logic
│   ├── models/     # Data models
│   ├── services/   # Business services
│   └── utils/      # Utility functions
├── static/         # Static files
├── templates/      # Template files
├── tests/          # Test files
├── app.py         # Main application entry point
├── config.py      # Configuration
└── requirements.txt # Project dependencies
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe for pose estimation
- OpenCV for image processing
- FastAPI for the web framework 