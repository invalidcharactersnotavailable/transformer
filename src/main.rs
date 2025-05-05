use std::fs;
use std::env;
use std::collections::HashMap;
use serde::Deserialize;
use burn::tensor::Tensor;
use burn::nn::{Module, Transformer, TransformerConfig};
use burn::optim::{Adam, Optimizer};
use burn::data::dataloader::DataLoader;
use std::fs::File;
use std::io::{Write, BufWriter};

#[derive(Deserialize)]
struct Config {
    model_dim: usize,
    num_heads: usize,
    num_layers: usize,
    feedforward_dim: usize,
    learning_rate: f32,
    batch_size: usize,
    epochs: usize,
    gradient_clipping: f32,
    warmup_steps: usize,
    vocab_size: usize,
    data_dir: String,
    log_dir: String,
    checkpoint_dir: String,
}

fn main() {
    // Parse command-line arguments
    let args: Vec<String> = env::args().collect();
    let mode = args.iter().find_map(|arg| {
        if arg.starts_with("--mode=") {
            Some(arg.trim_start_matches("--mode=").to_string())
        } else {
            None
        }
    }).unwrap_or_else(|| "train".to_string()); // Default to "train" mode

    // Load configuration from file
    let config: Config = load_config("./config.json");

    match mode.as_str() {
        "train" => {
            // Load and preprocess data
            let data = load_txt_files(&config.data_dir);
            let (tokenizer, tokenized_data) = tokenize_data(&data, config.vocab_size);

            // Define transformer model
            let transformer_config = TransformerConfig::new(
                config.model_dim,
                config.num_heads,
                config.num_layers,
                config.feedforward_dim,
            );
            let mut model = Transformer::new(transformer_config);

            // Define optimizer with learning rate scheduling
            let mut optimizer = Adam::new(config.learning_rate);
            let mut scheduler = LearningRateScheduler::new(config.learning_rate, config.warmup_steps);

            // Train the model
            train(
                &mut model,
                &mut optimizer,
                &mut scheduler,
                tokenized_data,
                config.epochs,
                config.batch_size,
                config.gradient_clipping,
                &config.log_dir,
                &config.checkpoint_dir,
            );
        }
        "chat" => {
            // Perform inference in chat mode
            let transformer_config = TransformerConfig::new(
                config.model_dim,
                config.num_heads,
                config.num_layers,
                config.feedforward_dim,
            );
            let mut model = Transformer::new(transformer_config);

            // Load tokenizer and model checkpoint
            let tokenizer = load_tokenizer(&config.checkpoint_dir);
            load_checkpoint(&mut model, &config.checkpoint_dir);

            println!("Chat mode activated. Type your input:");
            let mut input = String::new();
            while let Ok(_) = std::io::stdin().read_line(&mut input) {
                let output = infer(&model, &tokenizer, input.trim());
                println!("Response: {}", output);
                input.clear();
            }
        }
        _ => {
            eprintln!("Unknown mode: {}. Use --mode=train or --mode=chat.", mode);
        }
    }
}

fn load_config(path: &str) -> Config {
    let config_data = fs::read_to_string(path).expect("Failed to read config file");
    serde_json::from_str(&config_data).expect("Failed to parse config file")
}

fn load_txt_files(folder: &str) -> Vec<String> {
    let mut data = Vec::new();
    for entry in fs::read_dir(folder).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().unwrap_or_default() == "txt" {
            let content = fs::read_to_string(path).unwrap();
            data.push(content);
        }
    }
    data
}

fn tokenize_data(data: &[String], vocab_size: usize) -> (Tokenizer, Vec<Tensor<f32>>) {
    let tokenizer = Tokenizer::new(data, vocab_size);
    let tokenized_data = data
        .iter()
        .map(|text| tokenizer.tokenize(text))
        .collect();
    (tokenizer, tokenized_data)
}

fn train(
    model: &mut Transformer,
    optimizer: &mut Adam,
    scheduler: &mut LearningRateScheduler,
    data: Vec<Tensor<f32>>,
    epochs: usize,
    batch_size: usize,
    gradient_clipping: f32,
    log_dir: &str,
    checkpoint_dir: &str,
) {
    let log_file = File::create(format!("{}/training.log", log_dir)).expect("Failed to create log file");
    let mut log_writer = BufWriter::new(log_file);

    // Split data into batches with dynamic padding
    let batches: Vec<Vec<Tensor<f32>>> = data
        .chunks(batch_size)
        .map(|chunk| chunk.to_vec())
        .collect();

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;
        let mut correct_predictions = 0;
        let mut total_predictions = 0;

        for (batch_idx, batch) in batches.iter().enumerate() {
            // Forward pass
            let inputs = Tensor::stack(batch, 0); // Stack tensors in the batch
            let outputs = model.forward(&inputs);

            // Calculate loss (cross-entropy loss)
            let targets = inputs.clone(); // Placeholder: using inputs as targets
            let loss = cross_entropy_loss(&outputs, &targets);

            // Backward pass and optimization step
            optimizer.step_with_clipping(&loss, gradient_clipping);

            // Update learning rate
            scheduler.step(optimizer);

            // Calculate accuracy
            let predictions = outputs.argmax(1); // Get predicted classes
            let targets_classes = targets.argmax(1); // Get target classes
            correct_predictions += predictions.eq(&targets_classes).sum().to_scalar::<usize>();
            total_predictions += predictions.shape()[0];

            // Accumulate loss for logging
            epoch_loss += loss.to_scalar();

            writeln!(
                log_writer,
                "Epoch {}/{}, Batch {}/{}, Loss: {:.4}, Accuracy: {:.2}%",
                epoch + 1,
                epochs,
                batch_idx + 1,
                batches.len(),
                loss.to_scalar(),
                (correct_predictions as f32 / total_predictions as f32) * 100.0
            ).expect("Failed to write to log file");
        }

        let epoch_accuracy = (correct_predictions as f32 / total_predictions as f32) * 100.0;
        println!(
            "Epoch {}/{} completed. Average Loss: {:.4}, Accuracy: {:.2}%",
            epoch + 1,
            epochs,
            epoch_loss / batches.len() as f32,
            epoch_accuracy
        );

        // Save checkpoint
        save_checkpoint(model, checkpoint_dir, epoch);
    }
}

fn cross_entropy_loss(outputs: &Tensor<f32>, targets: &Tensor<f32>) -> Tensor<f32> {
    // Cross-entropy loss implementation
    let log_probs = outputs.log_softmax(1); // Apply log softmax
    let targets_one_hot = targets.one_hot(outputs.shape()[1]); // Convert targets to one-hot encoding
    -(targets_one_hot * log_probs).sum(1).mean()
}

struct LearningRateScheduler {
    base_lr: f32,
    warmup_steps: usize,
    step_count: usize,
}

impl LearningRateScheduler {
    fn new(base_lr: f32, warmup_steps: usize) -> Self {
        Self {
            base_lr,
            warmup_steps,
            step_count: 0,
        }
    }

    fn step(&mut self, optimizer: &mut Adam) {
        self.step_count += 1;
        let lr = if self.step_count < self.warmup_steps {
            self.base_lr * (self.step_count as f32 / self.warmup_steps as f32)
        } else {
            self.base_lr
        };
        optimizer.set_learning_rate(lr);
        println!("Learning rate updated to: {:.6}", lr);
    }
}

fn infer(model: &Transformer, tokenizer: &Tokenizer, input: &str) -> String {
    let tokenized_input = tokenizer.tokenize(input);
    let input_tensor = Tensor::from_data(tokenized_input);
    let output_tensor = model.forward(&input_tensor);
    tokenizer.detokenize(output_tensor.to_vec())
}

fn save_checkpoint(model: &Transformer, checkpoint_dir: &str, epoch: usize) {
    let path = format!("{}/model_epoch_{}.ckpt", checkpoint_dir, epoch);
    model.save(&path).expect("Failed to save checkpoint");
}

fn load_checkpoint(model: &mut Transformer, checkpoint_dir: &str) {
    let path = format!("{}/model_latest.ckpt", checkpoint_dir);
    model.load(&path).expect("Failed to load checkpoint");
}

fn load_tokenizer(checkpoint_dir: &str) -> Tokenizer {
    let path = format!("{}/tokenizer.json", checkpoint_dir);
    Tokenizer::load(&path).expect("Failed to load tokenizer")
}

struct Tokenizer {
    vocab: HashMap<String, usize>,
}

impl Tokenizer {
    fn new(data: &[String], vocab_size: usize) -> Self {
        let mut vocab = HashMap::new();
        let mut word_counts = HashMap::new();

        // Count word frequencies
        for text in data {
            for word in text.split_whitespace() {
                *word_counts.entry(word.to_string()).or_insert(0) += 1;
            }
        }

        // Sort by frequency and take the top `vocab_size` words
        let mut sorted_words: Vec<_> = word_counts.into_iter().collect();
        sorted_words.sort_by(|a, b| b.1.cmp(&a.1));
        for (idx, (word, _)) in sorted_words.into_iter().take(vocab_size).enumerate() {
            vocab.insert(word, idx);
        }

        Self { vocab }
    }

    fn tokenize(&self, text: &str) -> Vec<f32> {
        text.split_whitespace()
            .map(|word| *self.vocab.get(word).unwrap_or(&0) as f32)
            .collect()
    }

    fn detokenize(&self, tokens: Vec<f32>) -> String {
        let reverse_vocab: HashMap<_, _> = self.vocab.iter().map(|(k, v)| (*v, k)).collect();
        tokens
            .into_iter()
            .map(|token| reverse_vocab.get(&(token as usize)).unwrap_or(&"<UNK>").to_string())
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn save(&self, path: &str) {
        let serialized_vocab = serde_json::to_string(&self.vocab).expect("Failed to serialize tokenizer");
        fs::write(path, serialized_vocab).expect("Failed to save tokenizer");
    }

    fn load(path: &str) -> Result<Self, std::io::Error> {
        let serialized_vocab = fs::read_to_string(path)?;
        let vocab: HashMap<String, usize> = serde_json::from_str(&serialized_vocab)?;
        Ok(Self { vocab })
    }
}
