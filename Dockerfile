FROM python:3.10-slim
WORKDIR /app
VOLUME /app/data
SHELL [ "/bin/bash", "-c" ]
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python3","/app/make_submission.py"]

COPY . /app

RUN python3 -m venv venv && \
    source venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    python -c "from torchvision.models import resnet50, ResNet50_Weights; resnet50(ResNet50_Weights.IMAGENET1K_V2)" && \
    chmod +x /app/entrypoint.sh /app/baseline.py /app/make_submission.py
