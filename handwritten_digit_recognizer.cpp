#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <string>



void randomize_parameters(
	std::vector<float>& weights,
	std::vector<float>& biases,
	int hidden_layers_neurons_count,
	int input_layer_neurons_count,
	int hidden_layers_count,
	int output_layer_neurons_count) {

	//Random number generator initialization
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> weight_distribution(-1.0f, 1.0f);
	std::uniform_real_distribution<float> bias_distribution(-1.0, 1.0);
	
	// Generate random weights
	std::ofstream weights_file;
	weights_file.open("weights.csv");
	for (int i = 0; i < hidden_layers_neurons_count * input_layer_neurons_count + (hidden_layers_count - 1) * (hidden_layers_neurons_count * hidden_layers_neurons_count) + hidden_layers_neurons_count * output_layer_neurons_count; i++) {
		weights_file << weight_distribution(gen) << "\n";
	}

	weights_file.close();

	// Generate random biases
	std::ofstream biases_file;
	biases_file.open("biases.csv");
	for (int i = 0; i < hidden_layers_neurons_count * hidden_layers_count + output_layer_neurons_count; i++) {
		biases_file << bias_distribution(gen) << "\n";
	}

	biases_file.close();
}




std::vector<float> read_image(
	int image_resolution,
	std::ifstream& images,
	std::vector<float>& input_layer_neurons,
	int input_layer_neurons_count,
	int image_index) {

	// Seek to the correct image position (16-byte header + image size * index)
	images.seekg(16 + image_resolution * image_resolution * image_index, std::ios::beg);

	// Read the entire image
	for (int i = 0; i < input_layer_neurons_count; i++) {
		unsigned char pixel;
		images.read(reinterpret_cast<char*>(&pixel), 1);
		input_layer_neurons[i] = static_cast<float>(pixel) / 255.0f;  // Normalize to [0, 1]
	}

	return input_layer_neurons;
}





void read_label(
	std::ifstream& labels,
	std::vector<int>& expected_output_layer_neurons,
	int image_index) {

	// Seek to the correct label position (8-byte header + 1 byte per label)
	labels.seekg(8 + image_index, std::ios::beg);

	// Read the label
	unsigned char label;
	labels.read(reinterpret_cast<char*>(&label), 1);

	// Set the corresponding index to 1 (one-hot encoding)
	expected_output_layer_neurons[label] = 1;
}





void print_image(
	int image_resolution,
	std::vector<float>& input_layer_neurons) {

	// Print the first image
	for (int i = 0; i < image_resolution; i++) {
		for (int j = 0; j < image_resolution; j++) {
			if (input_layer_neurons[i * image_resolution + j] == 0) {
				std::cout << "  ";
			}

			else {
				std::cout << "##";
			}

		}
		std::cout << std::endl;
	}
}







void feedforward(
	int hidden_layers_neurons_count,
	int input_layer_neurons_count,
	std::vector<float>& hidden_layer_neurons,
	std::vector<float>& input_layer_neurons,
	std::vector<float>& weights,
	std::vector<float>& biases, int hidden_layers_count,
	std::vector<float>& output_layer_neurons,
	std::vector<float>& normalized_output_layer_neurons,
	int output_layer_neurons_count) {

	// First hidden layer neurons calculation
	for (int i = 0; i < hidden_layers_neurons_count; i++) {
		for (int j = 0; j < input_layer_neurons_count; j++) {
			hidden_layer_neurons[i] += input_layer_neurons[j] * weights[i * input_layer_neurons_count + j];
		}
		hidden_layer_neurons[i] += biases[i];
		hidden_layer_neurons[i] = 1 / (1 + exp(-hidden_layer_neurons[i]));
	}

	// Hidden layers neurons calculation
	for (int i = 0; i < hidden_layers_count - 1; i++) {
		for (int j = 0; j < hidden_layers_neurons_count; j++) {
			for (int k = 0; k < hidden_layers_neurons_count; k++) {
				hidden_layer_neurons[(i + 1) * hidden_layers_neurons_count + j] += hidden_layer_neurons[i * hidden_layers_neurons_count + j] * weights[input_layer_neurons_count * hidden_layers_neurons_count + i * (hidden_layers_neurons_count * hidden_layers_neurons_count) + j * hidden_layers_neurons_count + k];
			}
			hidden_layer_neurons[(i + 1) * hidden_layers_neurons_count + j] += biases[(i + 1) * hidden_layers_neurons_count + j];
			hidden_layer_neurons[(i + 1) * hidden_layers_neurons_count + j] = 1 / (1 + exp(-hidden_layer_neurons[(i + 1) * hidden_layers_neurons_count + j]));
		}
	}

	//Output layer neurons calculation
	for (int i = 0; i < output_layer_neurons_count; i++) {
		for (int j = 0; j < hidden_layers_neurons_count; j++) {
			output_layer_neurons[i] += hidden_layer_neurons[(hidden_layers_count - 1) * hidden_layers_neurons_count + j] * weights[input_layer_neurons_count * hidden_layers_neurons_count + (hidden_layers_count - 1) * (hidden_layers_neurons_count * hidden_layers_neurons_count) + i * hidden_layers_neurons_count + j];
		}
		output_layer_neurons[i] += biases[hidden_layers_count * hidden_layers_neurons_count + i];
	}

	// Softmax
	float max_output = *std::max_element(output_layer_neurons.begin(), output_layer_neurons.end());
	float sum = 0.0f;
	for (int i = 0; i < output_layer_neurons_count; i++) {
		normalized_output_layer_neurons[i] = exp(output_layer_neurons[i] - max_output);
		sum += normalized_output_layer_neurons[i];
	}

	for (int i = 0; i < output_layer_neurons_count; i++) {
		normalized_output_layer_neurons[i] /= sum;
	}
}



float compute_cost(
	int output_layer_neurons_count,
	std::vector<int> expected_output_layer_neurons,
	std::vector<float> normalized_output_layer_neurons) {

	// Cross-entropy loss
	float cost = 0.0f;
	for (int i = 0; i < output_layer_neurons_count; i++) {
		cost += -expected_output_layer_neurons[i] * log(normalized_output_layer_neurons[i]);
	}
	std::cout << "Cost: " << cost << std::endl;

	// Print the output layer neurons
	for (int i = 0; i < output_layer_neurons_count; i++) {
		std::cout << i << " : " << normalized_output_layer_neurons[i] << std::endl;
	}

	return cost;
}





void train(
	int output_layer_neurons_count,
	std::vector<float>& output_layer_neurons_error,
	std::vector<float>& normalized_output_layer_neurons,
	std::vector<int>& expected_output_layer_neurons,
	int hidden_layers_neurons_count,
	std::vector<float>& weights_sensitivity,
	int input_layer_neurons_count,
	int hidden_layers_count,
	std::vector<float>& hidden_layers_neurons,
	std::vector<float>& biases_sensitivity,
	std::vector<float>& hidden_layers_neurons_sensitivity,
	std::vector<float>& weights,
	std::vector<float>& input_layer_neurons,
	float learning_rate,
	std::vector<float>& biases,
	std::vector<float>& weights_sensitivity_sum,
	std::vector<float>& biases_sensitivity_sum) {

	// Compute cost for each output layer neuron
	for (int i = 0; i < output_layer_neurons_count; i++) {
		output_layer_neurons_error[i] = normalized_output_layer_neurons[i] - expected_output_layer_neurons[i];
	}


	// Output layer weights
	for (int i = 0; i < output_layer_neurons_count; i++) {
		for (int j = 0; j < hidden_layers_neurons_count; j++) {
			weights_sensitivity[input_layer_neurons_count * hidden_layers_neurons_count + (hidden_layers_count - 1) * hidden_layers_neurons_count * hidden_layers_neurons_count + i * hidden_layers_neurons_count + j] = hidden_layers_neurons[(hidden_layers_count - 1) * hidden_layers_neurons_count + j] * output_layer_neurons_error[i];
		}
	}

	// Output layer biases
	for (int i = 0; i < output_layer_neurons_count; i++) {
		biases_sensitivity[hidden_layers_neurons_count * hidden_layers_count + i] = output_layer_neurons_error[i];
	}

	// Last hidden layer neurons
	for (int i = 0; i < hidden_layers_neurons_count; i++) {
		for (int j = 0; j < output_layer_neurons_count; j++) {
			hidden_layers_neurons_sensitivity[(hidden_layers_count - 1) * hidden_layers_neurons_count + i] += output_layer_neurons_error[j] * weights[input_layer_neurons_count * hidden_layers_neurons_count + (hidden_layers_count - 1) * hidden_layers_neurons_count * hidden_layers_neurons_count + j * hidden_layers_neurons_count + i];
		}
		hidden_layers_neurons_sensitivity[(hidden_layers_count - 1) * hidden_layers_neurons_count + i] *= hidden_layers_neurons[(hidden_layers_count - 1) * hidden_layers_neurons_count + i] * (1 - hidden_layers_neurons[(hidden_layers_count - 1) * hidden_layers_neurons_count + i]);
	}

	// Hidden layers neurons
	for (int i = 0; i < (hidden_layers_count - 1); i++) {
		for (int j = 0; j < hidden_layers_neurons_count; j++) {
			for (int k = 0; k < hidden_layers_neurons_count; k++) {
				hidden_layers_neurons_sensitivity[(hidden_layers_count - 2 - i) * hidden_layers_neurons_count + j] += hidden_layers_neurons_sensitivity[(hidden_layers_count - 1 - i) * hidden_layers_neurons_count + k] * weights[input_layer_neurons_count * hidden_layers_neurons_count + (hidden_layers_count - 2 - i) * hidden_layers_neurons_count * hidden_layers_neurons_count + j * hidden_layers_neurons_count + k];
			}
			hidden_layers_neurons_sensitivity[(hidden_layers_count - 2 - i) * hidden_layers_neurons_count + j] *= hidden_layers_neurons[(hidden_layers_count - 2 - i) * hidden_layers_neurons_count + j] * (1 - hidden_layers_neurons[(hidden_layers_count - 2 - i) * hidden_layers_neurons_count + j]);
		}
	}

	// Hidden layers weights
	for (int i = 0; i < (hidden_layers_count - 1); i++) {
		for (int j = 0; j < hidden_layers_neurons_count; j++) {
			for (int k = 0; k < hidden_layers_neurons_count; k++) {
				weights_sensitivity[input_layer_neurons_count * hidden_layers_neurons_count + (hidden_layers_count - 2 - i) * hidden_layers_neurons_count * hidden_layers_neurons_count + j * hidden_layers_neurons_count + k] = hidden_layers_neurons_sensitivity[(hidden_layers_count - 2 - i) * hidden_layers_neurons_count + k] * hidden_layers_neurons[(hidden_layers_count - 2 - i) * hidden_layers_neurons_count + j];
			}
		}
	}

	// Hidden layers biases
	for (int i = 0; i < hidden_layers_count; i++) {
		for (int j = 0; j < hidden_layers_neurons_count; j++) {
			biases_sensitivity[(hidden_layers_count - 1 - i) * hidden_layers_neurons_count + j] = hidden_layers_neurons_sensitivity[(hidden_layers_count - 1 - i) * hidden_layers_neurons_count + j];
		}
	}

	// Input layer weights
	for (int i = 0; i < hidden_layers_neurons_count; i++) {
		for (int j = 0; j < input_layer_neurons_count; j++) {
			weights_sensitivity[i * input_layer_neurons_count + j] = hidden_layers_neurons_sensitivity[i] * input_layer_neurons[j];
		}
	}


	// Add the weights sensitivity to the sum
	for (int i = 0; i < hidden_layers_neurons_count * input_layer_neurons_count + (hidden_layers_count - 1) * (hidden_layers_neurons_count * hidden_layers_neurons_count) + hidden_layers_neurons_count * output_layer_neurons_count; i++) {
		weights_sensitivity_sum[i] += weights_sensitivity[i];
	}


	// Add the biases sensitivity to the sum
	for (int i = 0; i < hidden_layers_neurons_count * hidden_layers_count + output_layer_neurons_count; i++) {
		biases_sensitivity_sum[i] += biases_sensitivity[i];
	}
}


void update_parameters(
	std::vector<float>& weights_sensitivity_sum,
	int batch_size,
	std::vector<float>& biases_sensitivity_sum,
	int hidden_layers_neurons_count,
	int input_layer_neurons_count,
	int hidden_layers_count,
	int output_layer_neurons_count,
	std::vector<float>& weights,
	float learning_rate,
	std::vector<float>& weights_sensitivity,
	std::vector<float>& biases,
	std::vector<float>& biases_sensitivity) {


	// Average the weights sensitivity sum
	for (int i = 0; i < hidden_layers_neurons_count * input_layer_neurons_count + (hidden_layers_count - 1) * (hidden_layers_neurons_count * hidden_layers_neurons_count) + hidden_layers_neurons_count * output_layer_neurons_count; i++) {
		weights_sensitivity_sum[i] /= batch_size;
	}

	// Average the biases sensitivity sum
	for (int i = 0; i < hidden_layers_neurons_count * hidden_layers_count + output_layer_neurons_count; i++) {
		biases_sensitivity_sum[i] /= batch_size;
	}



	// Update weights
	for (int i = 0; i < hidden_layers_neurons_count * input_layer_neurons_count + (hidden_layers_count - 1) * (hidden_layers_neurons_count * hidden_layers_neurons_count) + hidden_layers_neurons_count * output_layer_neurons_count; i++) {
		weights[i] -= learning_rate * weights_sensitivity_sum[i];
	}


	// Update biases
	for (int i = 0; i < hidden_layers_neurons_count * hidden_layers_count + output_layer_neurons_count; i++) {
		biases[i] -= learning_rate * biases_sensitivity_sum[i];
	}
}





void update_parameters_files(
	int hidden_layers_neurons_count,
	int input_layer_neurons_count,
	int hidden_layers_count,
	int output_layer_neurons_count,
	std::vector<float>& weights,
	std::vector<float>& biases) {


	// Update weights file
	std::ofstream weights_file;
	weights_file.open("weights.csv");
	for (int i = 0; i < hidden_layers_neurons_count * input_layer_neurons_count + (hidden_layers_count - 1) * (hidden_layers_neurons_count * hidden_layers_neurons_count) + hidden_layers_neurons_count * output_layer_neurons_count; i++) {
		weights_file << weights[i] << "\n";
	}

	weights_file.close();


	// Update biases file
	std::ofstream biases_file;
	biases_file.open("biases.csv");
	for (int i = 0; i < hidden_layers_neurons_count * hidden_layers_count + output_layer_neurons_count; i++) {
		biases_file << biases[i] << "\n";
	}

	biases_file.close();
}




void reset_vectors(
	std::vector<float>& input_layer_neurons,
	std::vector<float>& hidden_layers_neurons,
	std::vector<float>& output_layer_neurons,
	std::vector<float>& normalized_output_layer_neurons,
	std::vector<float>& output_layer_neurons_error,
	std::vector<float>& weights_sensitivity,
	std::vector<float>& biases_sensitivity,
	std::vector<float>& hidden_layers_neurons_sensitivity,
	std::vector<int>& expected_output_layer_neurons) {
	
	// Set every vector values to 0
	std::fill(input_layer_neurons.begin(), input_layer_neurons.end(), 0.0f);
	std::fill(hidden_layers_neurons.begin(), hidden_layers_neurons.end(), 0.0f);
	std::fill(output_layer_neurons.begin(), output_layer_neurons.end(), 0.0f);
	std::fill(normalized_output_layer_neurons.begin(), normalized_output_layer_neurons.end(), 0.0f);
	std::fill(output_layer_neurons_error.begin(), output_layer_neurons_error.end(), 0.0f);
	std::fill(weights_sensitivity.begin(), weights_sensitivity.end(), 0.0f);
	std::fill(biases_sensitivity.begin(), biases_sensitivity.end(), 0.0f);
	std::fill(hidden_layers_neurons_sensitivity.begin(), hidden_layers_neurons_sensitivity.end(), 0.0f);
	std::fill(expected_output_layer_neurons.begin(), expected_output_layer_neurons.end(), 0);
}






int main() {


	// Constants initialization
	const int image_resolution = 28;
	const int input_layer_neurons_count = image_resolution * image_resolution;
	const int hidden_layers_neurons_count = 16;
	const int output_layer_neurons_count = 10;
	const int hidden_layers_count = 2;
	const float learning_rate = 0.1f;
	const int batch_size = 64;


	// Vectors initialization
	std::vector<float> input_layer_neurons(input_layer_neurons_count, 0.0f);
	std::vector<float> weights(hidden_layers_neurons_count * input_layer_neurons_count + (hidden_layers_count - 1) * (hidden_layers_neurons_count * hidden_layers_neurons_count) + hidden_layers_neurons_count * output_layer_neurons_count, 0.0f);
	std::vector<float> hidden_layers_neurons(hidden_layers_neurons_count * hidden_layers_count, 0.0f);
	std::vector<float> biases(hidden_layers_neurons_count * hidden_layers_count + output_layer_neurons_count, 0.0f);
	std::vector<float> output_layer_neurons(output_layer_neurons_count, 0.0f);
	std::vector<float> normalized_output_layer_neurons(output_layer_neurons_count, 0.0f);
	std::vector<float> output_layer_neurons_error(output_layer_neurons_count, 0.0f);
	std::vector<float> weights_sensitivity(hidden_layers_neurons_count * input_layer_neurons_count + (hidden_layers_count - 1) * (hidden_layers_neurons_count * hidden_layers_neurons_count) + hidden_layers_neurons_count * output_layer_neurons_count, 0.0f);
	std::vector<float> biases_sensitivity(hidden_layers_neurons_count * hidden_layers_count + output_layer_neurons_count, 0.0f);
	std::vector<float> hidden_layers_neurons_sensitivity((hidden_layers_count - 1) * hidden_layers_neurons_count * hidden_layers_neurons_count, 0.0f);
	std::vector<float> weights_sensitivity_sum(hidden_layers_neurons_count * input_layer_neurons_count + (hidden_layers_count - 1) * (hidden_layers_neurons_count * hidden_layers_neurons_count) + hidden_layers_neurons_count * output_layer_neurons_count, 0.0f);
	std::vector<float> biases_sensitivity_sum(hidden_layers_neurons_count * input_layer_neurons_count + (hidden_layers_count - 1) * (hidden_layers_neurons_count * hidden_layers_neurons_count) + hidden_layers_neurons_count * output_layer_neurons_count, 0.0f);
	std::vector<int> expected_output_layer_neurons(output_layer_neurons_count, 0);


	// Dataset loading
	std::ifstream images("t10k-images.idx3-ubyte", std::ios::binary);
	if (!images) {
		std::cerr << "Error opening images" << std::endl;
		return 1;
	}
	images.seekg(16, std::ios::beg);



	std::ifstream labels("t10k-labels.idx1-ubyte", std::ios::binary);
	if (!labels) {
		std::cerr << "Error opening labels" << std::endl;
		return 1;
	}
	labels.seekg(8, std::ios::beg);


	// Open the weights file
	std::ifstream weights_data("weights.csv");
	if (!weights_data) {
		std::cerr << "Error opening file" << std::endl;
		return 1;
	}


	// Read the weights
	std::string line;
	for (int i = 0; i < hidden_layers_neurons_count * input_layer_neurons_count + (hidden_layers_count - 1) * (hidden_layers_neurons_count * hidden_layers_neurons_count) + hidden_layers_neurons_count * output_layer_neurons_count; i++) {
		std::getline(weights_data, line);
		weights[i] = std::stof(line);

	}



	// Open the biases file
	std::ifstream biases_data("biases.csv");
	if (!biases_data) {
		std::cerr << "Error opening file" << std::endl;
		return 1;
	}



	// Read the biases
	for (int i = 0; i < hidden_layers_neurons_count * hidden_layers_count + output_layer_neurons_count; i++) {
		std::getline(biases_data, line);
		biases[i] = std::stof(line);
	}





	int image_index = 0;


	for (int i = 0; i < 100; i++) {
		for (int j = 0; j < batch_size; j++) {

			read_image(image_resolution, images, input_layer_neurons, input_layer_neurons_count, image_index);

			read_label(labels, expected_output_layer_neurons, image_index);

			//print_image(image_resolution, input_layer_neurons);

			feedforward(hidden_layers_neurons_count, input_layer_neurons_count, hidden_layers_neurons, input_layer_neurons, weights, biases, hidden_layers_count, output_layer_neurons, normalized_output_layer_neurons, output_layer_neurons_count);

			compute_cost(output_layer_neurons_count, expected_output_layer_neurons, normalized_output_layer_neurons);

			train(output_layer_neurons_count, output_layer_neurons_error, normalized_output_layer_neurons, expected_output_layer_neurons, hidden_layers_neurons_count, weights_sensitivity, input_layer_neurons_count, hidden_layers_count, hidden_layers_neurons, biases_sensitivity, hidden_layers_neurons_sensitivity, weights, input_layer_neurons, learning_rate, biases, weights_sensitivity_sum, biases_sensitivity_sum);

			std::cout << "Predicted number : " << std::distance(normalized_output_layer_neurons.begin(), std::max_element(normalized_output_layer_neurons.begin(), normalized_output_layer_neurons.end())) << "\n";

			reset_vectors(input_layer_neurons, hidden_layers_neurons, output_layer_neurons, normalized_output_layer_neurons, output_layer_neurons_error, weights_sensitivity, biases_sensitivity, hidden_layers_neurons_sensitivity, expected_output_layer_neurons);


			image_index++;
		}

		update_parameters(weights_sensitivity_sum, batch_size, biases_sensitivity_sum, hidden_layers_neurons_count, input_layer_neurons_count, hidden_layers_count, output_layer_neurons_count, weights, learning_rate, weights_sensitivity, biases, biases_sensitivity);

		update_parameters_files(hidden_layers_neurons_count, input_layer_neurons_count, hidden_layers_count, output_layer_neurons_count, weights, biases);
	}

	return 0;
}