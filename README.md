# Safety-Aware Small Language Models for Automotive Applications (still under construction)

## Overview
This project implements a framework for training and deploying Small Language Models (SLMs) in automotive systems with built-in safety guarantees. The framework focuses on efficient command processing while maintaining strict safety compliance and real-time performance requirements.

## Key Features
- Multiple SLM architectures support (Phi-2, BERT-small, DistilBERT, TinyBERT, ALBERT)
- Safety-aware model adaptation using LoRA (Low-Rank Adaptation)
- Real-time safety verification system
- Resource-efficient implementation
- Comprehensive evaluation framework
- Command generation and validation
- Interactive visualization tools

## Project Structure
```
automotive_slm/
├── config/                  # Configuration files
│   ├── model_config.py
│   ├── training_config.yaml
│   ├── model_selection_config.yaml
│   └── command_hierarchy.json
├── data/                    # Data handling
│   ├── command_generator.py
│   ├── automotive_dataset.py
│   └── data_processor.py
├── models/                  # Model implementations
│   ├── model_factory.py
│   └── automotive_adapter.py
├── safety/                  # Safety components
│   ├── safety_checker.py
│   └── safety_rules.py
├── training/               # Training components
│   ├── trainer.py
│   └── optimization.py
├── evaluation/             # Evaluation tools
│   ├── evaluator.py
│   └── metrics.py
├── visualization/          # Visualization tools
│   └── performance_plots.py
└── scripts/                # Executable scripts
    ├── train.py
    ├── evaluate.py
    ├── select_model.py
    └── generate_data.py
```

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 8GB RAM minimum

### Setup (Still under test)
1. Clone the repository:
```bash
git clone https://github.com/yourusername/automotive-slm.git
cd automotive-slm
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Training Data
Generate synthetic automotive commands for training:
```bash
python scripts/generate_data.py \
    --config config/training_config.yaml \
    --output-dir data/ \
    --num-samples 10000
```

### 2. Select Model
Compare and select the best model for your use case:
```bash
python scripts/select_model.py \
    --config config/model_selection_config.yaml \
    --output-dir results/
```

### 3. Train Model
Train the selected model with safety-aware features:
```bash
python scripts/train.py \
    --config config/training_config.yaml \
    --model-key phi-2 \
    --output-dir models/
```

### 4. Evaluate Model
Evaluate the trained model's performance:
```bash
python scripts/evaluate.py \
    --model-path models/best_model \
    --config config/training_config.yaml \
    --output-dir evaluation/
```

## Model Architectures

### Supported Models
1. **Phi-2 (2.7B parameters)**
   - Base architecture for rich language understanding
   - Efficient adaptation using LoRA

2. **BERT-small (14M parameters)**
   - Compact BERT variant
   - Efficient for specific tasks

3. **DistilBERT (66M parameters)**
   - Knowledge-distilled BERT
   - Good balance of size and performance

4. **TinyBERT (14.5M parameters)**
   - Highly compressed BERT
   - Optimized for edge devices

5. **ALBERT (12M parameters)**
   - Parameter-efficient architecture
   - Cross-layer parameter sharing

## Safety Features

### Command Validation
- Parameter range checking
- Context-aware validation
- Real-time safety verification
- Violation detection and logging

### Safety Rules
- Speed limits
- Temperature ranges
- Context-dependent constraints
- Emergency override handling

## Performance Metrics

The framework evaluates models on:
- Command understanding accuracy
- Safety compliance rate
- Response latency
- Memory usage
- GPU utilization

## Visualization

Built-in visualization tools for:
- Training progress
- Performance metrics
- Resource utilization
- Safety violations
- Model comparisons

## Contributing

.
.
.
.

## Testing Individual Components

.

```bash
# Test data components
python data/command_generator.py
python data/automotive_dataset.py

# Test model components
python models/model_factory.py
python models/automotive_adapter.py

# Test safety components
python safety/safety_checker.py
```

## Configuration

### Model Configuration
Edit `config/training_config.yaml` to modify:
- Model parameters
- Training settings
- Safety rules
- Resource limits

### Command Configuration
Edit `config/command_hierarchy.json` to modify:
- Command types
- Parameter ranges
- Safety constraints
- Context rules

## Logs and Output

All scripts generate detailed logs in their respective output directories:
- Training logs: `models/logs/`
- Evaluation results: `evaluation/`
- Model selection results: `results/`
- Generated data: `data/`

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Transformers library by Hugging Face
- LoRA implementation
- PyTorch framework

## Contact
  akramsalim9@gmail.com
Project Link: [https://github.com/akramsalim/automotive-slm](https://github.com/akramsalim/automotive-slm)
