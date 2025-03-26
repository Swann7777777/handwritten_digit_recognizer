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

	std::cout << "Generated random weights" << std::endl;

	// Generate random biases
	std::ofstream biases_file;
	biases_file.open("biases.csv");
	for (int i = 0; i < hidden_layers_neurons_count * hidden_layers_count + output_layer_neurons_count; i++) {
		biases_file << bias_distribution(gen) << "\n";
	}

	biases_file.close();

	std::cout << "Generated random biases" << std::endl;
}







std::vector<float> read_image(
	std::ifstream& images,
	std::vector<float>& input_layer_neurons,
	int input_layer_neurons_count) {

	// Read the image
	char byte;
	for (int i = 0; i < input_layer_neurons_count; i++) {
		images.get(byte);
		float value = static_cast<float>(static_cast<unsigned char>(byte));
		input_layer_neurons[i] = value;
	}

	std::cout << "Read the image" << std::endl;

	return input_layer_neurons;
}








std::vector<int> read_label(
	std::ifstream& labels,
	std::vector<int>& expected_output_layer_neurons) {

	// Read the label
	char label;
	labels.get(label);
	expected_output_layer_neurons[static_cast<unsigned>(label)] = 1;

	return expected_output_layer_neurons;
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

	std::cout << "Printed the image" << std::endl;

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
	std::vector<float> output_layer_neurons,
	std::vector<float> output_layer_neurons_error,
	std::vector<int> expected_output_layer_neurons) {

	// Compute cost for each output layer neuron
	for (int i = 0; i < output_layer_neurons_count; i++) {
		output_layer_neurons_error[i] = output_layer_neurons[i] - expected_output_layer_neurons[i];
	}


	// Compute the sensivity of each weight of the output layer




}








// Steps to train the network :
// 1. Feedforward an image
// 2. Calculate the cost for each neuron
// 3. Calculate the sensivity of the weights and biases of the output layer
// 4. Calculate the sensivity of the weights and biases of the hidden layers
// 5. Update the weights and biases
// 6. Repeat the process with a mini-batch of images
// 7. Average the weights and biases
// 8. Save the weights and biases





int main() {


	// Constants initialization
	const int image_resolution = 28;
	const int input_layer_neurons_count = image_resolution * image_resolution;
	const int hidden_layers_neurons_count = 16;
	const int output_layer_neurons_count = 10;
	const int hidden_layers_count = 2;
	const float learning_rate = 0.1f;

	std::cout << "Initialized constants" << std::endl;


	// Vectors initialization
	std::vector<float> input_layer_neurons(input_layer_neurons_count, 0.0f);
	std::vector<float> weights(hidden_layers_neurons_count * input_layer_neurons_count + (hidden_layers_count - 1) * (hidden_layers_neurons_count * hidden_layers_neurons_count) + hidden_layers_neurons_count * output_layer_neurons_count, 0.0f);
	std::vector<float> hidden_layer_neurons(hidden_layers_neurons_count * hidden_layers_count, 0.0f);
	std::vector<float> biases(hidden_layers_neurons_count * hidden_layers_count + output_layer_neurons_count, 0.0f);
	std::vector<float> output_layer_neurons(output_layer_neurons_count, 0.0f);
	std::vector<float> normalized_output_layer_neurons(output_layer_neurons_count, 0.0f);
	std::vector<float> output_layer_neurons_error(output_layer_neurons_count, 0.0f);
	std::vector<float> weights_sensivity(hidden_layers_neurons_count * input_layer_neurons_count + (hidden_layers_count - 1) * (hidden_layers_neurons_count * hidden_layers_neurons_count) + hidden_layers_neurons_count * output_layer_neurons_count, 0.0f);
	std::vector<float> hidden_layer_neurons_sensitivity((hidden_layers_count - 1) * hidden_layers_neurons_count * hidden_layers_neurons_count, 0.0f);
	std::vector<int> expected_output_layer_neurons(output_layer_neurons_count, 0);

	std::cout << "Initialized vectors" << std::endl;


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

	std::cout << "Loaded dataset" << std::endl;



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







	read_image(images, input_layer_neurons, input_layer_neurons_count);

	read_label(labels, expected_output_layer_neurons);

	print_image(image_resolution, input_layer_neurons);

	feedforward(hidden_layers_neurons_count, input_layer_neurons_count, hidden_layer_neurons, input_layer_neurons, weights, biases, hidden_layers_count, output_layer_neurons, normalized_output_layer_neurons, output_layer_neurons_count);

	compute_cost(output_layer_neurons_count, expected_output_layer_neurons, normalized_output_layer_neurons);








	// Compute cost for each output layer neuron
	for (int i = 0; i < output_layer_neurons_count; i++) {
		output_layer_neurons_error[i] = normalized_output_layer_neurons[i] - expected_output_layer_neurons[i];
	}


	// Output layer weights
	for (int i = 0; i < output_layer_neurons_count; i++) {
		for (int j = 0; j < hidden_layers_neurons_count; j++) {
			weights_sensivity[input_layer_neurons_count * hidden_layers_neurons_count + (hidden_layers_count - 1) * hidden_layers_neurons_count * hidden_layers_neurons_count + i * hidden_layers_neurons_count + j] = hidden_layer_neurons[(hidden_layers_count - 1) * hidden_layers_neurons_count + j] * output_layer_neurons_error[i];
		}
	}

	// Last hidden layer neurons
	for (int i = 0; i < hidden_layers_neurons_count; i++) {
		for (int j = 0; j < output_layer_neurons_count; j++) {
			hidden_layer_neurons_sensitivity[(hidden_layers_count - 1) * hidden_layers_neurons_count + i] += output_layer_neurons_error[j] * weights[input_layer_neurons_count * hidden_layers_neurons_count + (hidden_layers_count - 1) * hidden_layers_neurons_count * hidden_layers_neurons_count + i * output_layer_neurons_count + j];
		}
		hidden_layer_neurons_sensitivity[(hidden_layers_count - 1) * hidden_layers_neurons_count + i] *= hidden_layer_neurons[(hidden_layers_count - 1) * hidden_layers_neurons_count + i] * (1 - hidden_layer_neurons[(hidden_layers_count - 1) * hidden_layers_neurons_count + i]);
	}

	// Hidden layers neurons
	for (int i = 0; i < (hidden_layers_count - 1); i++) {
		for (int j = 0; j < hidden_layers_neurons_count; j++) {
			for (int k = 0; k < hidden_layers_neurons_count; k++) {
				hidden_layer_neurons_sensitivity[(hidden_layers_count - 2 - i) * hidden_layers_neurons_count + j] += hidden_layer_neurons_sensitivity[(hidden_layers_count - 1 - i) * hidden_layers_neurons_count + k] * weights[input_layer_neurons_count * hidden_layers_neurons_count + (hidden_layers_count - 2 - i) * hidden_layers_neurons_count * hidden_layers_neurons_count + j * hidden_layers_neurons_count + k];
			}
			hidden_layer_neurons_sensitivity[(hidden_layers_count - 2 - i) * hidden_layers_neurons_count + j] *= hidden_layer_neurons[(hidden_layers_count - 2 - i) * hidden_layers_neurons_count + j] * (1 - hidden_layer_neurons[(hidden_layers_count - 2 - i) * hidden_layers_neurons_count + j]);
		}
	}

	// Hidden layers weights and neurons
	for (int i = 0; i < (hidden_layers_count - 1); i++) {
		for (int j = 0; j < hidden_layers_neurons_count; j++) {
			for (int k = 0; k < hidden_layers_neurons_count; k++) {
				weights_sensivity[input_layer_neurons_count * hidden_layers_neurons_count + (hidden_layers_count - 2 - i) * hidden_layers_neurons_count * hidden_layers_neurons_count + j * hidden_layers_neurons_count + k] = hidden_layer_neurons_sensitivity[(hidden_layers_count - 2 - i) * hidden_layers_neurons_count + k] * hidden_layer_neurons[(hidden_layers_count - 2 - i) * hidden_layers_neurons_count + j];
			}
		}
	}


	for (int i = 0; i < hidden_layers_neurons_count * input_layer_neurons_count + (hidden_layers_count - 1) * (hidden_layers_neurons_count * hidden_layers_neurons_count) + hidden_layers_neurons_count * output_layer_neurons_count; i++) {
		weights[i] -= learning_rate * weights_sensivity[i];
	}


	std::ofstream weights_file;
	weights_file.open("weights.csv");
	for (int i = 0; i < hidden_layers_neurons_count * input_layer_neurons_count + (hidden_layers_count - 1) * (hidden_layers_neurons_count * hidden_layers_neurons_count) + hidden_layers_neurons_count * output_layer_neurons_count; i++) {
		weights_file << weights[i] << "\n";
	}

	weights_file.close();





	return 0;
}