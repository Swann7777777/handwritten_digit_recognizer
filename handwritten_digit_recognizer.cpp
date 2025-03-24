#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <string>

int main() {

	// Random number generator initialization
	//std::random_device rd;
	//std::mt19937 gen(rd());
	//std::uniform_real_distribution<float> dis_weight(-1.0f, 1.0f);
	//std::uniform_real_distribution<float> dis_bias(-1.0, 1.0);
	//std::cout << "Initialized random number generator" << std::endl;

	// Constants initialization
	const int image_resolution = 280;
	const int input_layer_neurons_count = image_resolution * image_resolution;
	const int hidden_layers_neurons_count = 16;
	const int output_layer_neurons_count = 10;
	const int hidden_layers_count = 2;
	std::cout << "Initialized constants" << std::endl;

	// Vectors initialization
	std::vector<float> input_layer_neurons(input_layer_neurons_count, 0.0f);
	std::vector<float> weights(hidden_layers_neurons_count * input_layer_neurons_count + (hidden_layers_count - 1) * (hidden_layers_neurons_count * hidden_layers_neurons_count) + hidden_layers_neurons_count * output_layer_neurons_count, 0.0f);
	std::vector<float> hidden_layer_neurons(hidden_layers_neurons_count, 0.0f);
	std::vector<float> biases(hidden_layers_neurons_count * hidden_layers_count + output_layer_neurons_count, 0.0f);
	std::vector<float> temporary_hidden_layer_neurons(hidden_layers_neurons_count, 0.0f);
	std::vector<float> output_layer_neurons(output_layer_neurons_count, 0.0f);
	std::vector<int> image_labels(output_layer_neurons_count, 0);
	std::cout << "Initialized vectors" << std::endl;

	// Dataset loading
	std::ifstream images("t10k-images.idx3-ubyte", std::ios::binary);
	if (!images) {
		std::cerr << "Error opening images" << std::endl;
		return 1;
	}

	std::ifstream labels("t10k-labels.idx1-ubyte", std::ios::binary);
	if (!labels) {
		std::cerr << "Error opening labels" << std::endl;
		return 1;
	}

	std::cout << "Loaded dataset" << std::endl;




	// Read the first image
	char byte;
	images.seekg(16 + 0 * (28 * 28), std::ios::beg);
	for (int i = 0; i < input_layer_neurons_count; i++) {
		images.get(byte);
		float value = static_cast<float>(static_cast<unsigned char>(byte));
		input_layer_neurons[i] = value;
	}

	labels.seekg(8 + 0, std::ios::beg);
	char label;
	labels.get(label);
	image_labels[static_cast<unsigned>(label)] = 1;


	std::cout << "Read the image" << std::endl;




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

	std::cout << "Printed the image" << std::endl;


	// Generate random weights
	//std::ofstream weights_file;
	//weights_file.open("weights.csv");
	//for (int i = 0; i < hidden_layers_neurons_count * input_layer_neurons_count + (hidden_layers_count - 1) * (hidden_layers_neurons_count * hidden_layers_neurons_count) + hidden_layers_neurons_count * output_layer_neurons_count; i++) {
		//weights_file << dis_weight(gen) << "\n";
	//}

	//weights_file.close();
	//std::cout << "Generated random weights" << std::endl;

	// Generate random biases

	//std::ofstream biases_file;
	//biases_file.open("biases.csv");
	//for (int i = 0; i < hidden_layers_neurons_count * hidden_layers_count + output_layer_neurons_count; i++) {
		//biases_file << dis_bias(gen) << "\n";
	//}

	//biases_file.close();
	//std::cout << "Generated random biases" << std::endl;


	// Open the weights file
	std::ifstream weights_data("weights.csv");
	if (!weights_data) {
		std::cerr << "Error opening file" << std::endl;
		return 1;
	}
	std::cout << "Opened the weights file" << std::endl;


	// Read the weights
	std::string line;
	for (int i = 0; i < hidden_layers_neurons_count * input_layer_neurons_count + (hidden_layers_count - 1) * (hidden_layers_neurons_count * hidden_layers_neurons_count) + hidden_layers_neurons_count * output_layer_neurons_count; i++) {
		std::getline(weights_data, line);
		weights[i] = std::stof(line);

	}

	std::cout << "Read the weights" << std::endl;


	// Open the biases file
	std::ifstream biases_data("biases.csv");
	if (!biases_data) {
		std::cerr << "Error opening file" << std::endl;
		return 1;
	}
	std::cout << "Opened the biases file" << std::endl;



	// Read the biases
	for (int i = 0; i < hidden_layers_neurons_count * hidden_layers_count + output_layer_neurons_count; i++) {
		std::getline(biases_data, line);
		biases[i] = std::stof(line);
	}
	std::cout << "Read the biases" << std::endl;


	// First hidden layer neurons calculation
	for (int i = 0; i < hidden_layers_neurons_count; i++) {
		for (int j = 0; j < input_layer_neurons_count; j++) {
			hidden_layer_neurons[i] += input_layer_neurons[j] * weights[i * input_layer_neurons_count + j];
		}
		hidden_layer_neurons[i] += biases[i];
	}




	// Hidden layers neurons calculation
	for (int i = 0; i < hidden_layers_count - 1; i++) {
		temporary_hidden_layer_neurons = hidden_layer_neurons;
		for (int j = 0; j < hidden_layers_neurons_count; j++) {
			hidden_layer_neurons[j] = 0.0f;
			temporary_hidden_layer_neurons[j] = 1 / (1 + exp(-temporary_hidden_layer_neurons[j]));
			for (int k = 0; k < hidden_layers_neurons_count; k++) {
				hidden_layer_neurons[j] += temporary_hidden_layer_neurons[j] * weights[input_layer_neurons_count * hidden_layers_neurons_count + i * (hidden_layers_neurons_count * hidden_layers_neurons_count) + j * hidden_layers_neurons_count + k];
			}
			hidden_layer_neurons[j] += biases[(i + 1) * hidden_layers_neurons_count + j];
		}
	}

	//Output layer neurons calculation
	for (int i = 0; i < output_layer_neurons_count; i++) {
		for (int j = 0; j < hidden_layers_neurons_count; j++) {
			output_layer_neurons[i] += hidden_layer_neurons[j] * weights[input_layer_neurons_count * hidden_layers_neurons_count + (hidden_layers_count - 1) * (hidden_layers_neurons_count * hidden_layers_neurons_count) + i * hidden_layers_neurons_count + j];
		}
		output_layer_neurons[i] += biases[hidden_layers_count * hidden_layers_neurons_count + i];
	}



	// Softmax
	float max_output = *std::max_element(output_layer_neurons.begin(), output_layer_neurons.end());
	float sum = 0.0f;
	for (int i = 0; i < output_layer_neurons_count; i++) {
		output_layer_neurons[i] = exp(output_layer_neurons[i] - max_output);
		sum += output_layer_neurons[i];
	}

	for (int i = 0; i < output_layer_neurons_count; i++) {
		output_layer_neurons[i] /= sum;
	}



	// Cross-entropy loss
	float cost = 0.0f;
	for (int i = 0; i < output_layer_neurons_count; i++) {
		cost += -image_labels[i] * log(output_layer_neurons[i]);
	}
	std::cout << "Cost: " << cost << std::endl;

	for (int i = 0; i < output_layer_neurons_count; i++) {
		output_layer_neurons[i] = std::round(output_layer_neurons[i] * 100.0f) / 100.0f;
	}
	// Print the output layer neurons

	for (int i = 0; i < output_layer_neurons_count; i++) {
		std::cout << i << " : " << output_layer_neurons[i] << std::endl;
	}


	return 0;
}