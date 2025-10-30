FROM python:3.13

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies needed to build some Python packages
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
	   build-essential \
	   git \
	   curl \
	   pkg-config \
	   libffi-dev \
	   libpq-dev \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel \
	&& pip install -r requirements.txt

# Copy the project files
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Default streamlit app - can be overridden at runtime with a different CMD
CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=8501"]

