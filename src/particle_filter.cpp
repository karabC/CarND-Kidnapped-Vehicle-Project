/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <float.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	/*** Initialization ***/

	/* Create Gaussian distribution for x, y, theta*/
	default_random_engine gen;
	normal_distribution<double> distribution_x(x, std[0]);
	normal_distribution<double> distribution_y(y, std[1]);
	normal_distribution<double> distribution_theta(theta, std[2]);

	/*Set the number of particles. */
	num_particles = 30;

	/* Sample the Particles from distributions*/
	for (int i = 0; i < num_particles; ++i) {
		double sample_x = distribution_x(gen);
		double sample_y = distribution_y(gen);
		double sample_theta = distribution_theta(gen);

		Particle temp;
		temp.id = i;
		temp.x = sample_x;
		temp.y = sample_y;
		temp.theta = sample_theta;
		temp.weight = 1;
		particles.push_back(temp);

		/* Set Weights to 1*/
		weights.push_back(1.0);

		// Print your samples to the terminal.
		// cout << "Initialization Sample " << i + 1 << ": " << sample_x << ", " << sample_y << " " << endl;
	}

	/* Mark Initialization done*/
	is_initialized = true;

	cout << "Initialization Done " << endl;
}

/* Add measurements to each particle and add random Gaussian noise. */
void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	cout << "Prediction Start " << endl;

	/* Create distributions of Gaussian noise for x, y, theta*/
	default_random_engine gen;
	normal_distribution<double> noise_x(0, std_pos[0]);
	normal_distribution<double> noise_y(0, std_pos[1]);
	normal_distribution<double> noise_theta(0, std_pos[2]);

	/* Set minimum Yaw for divide-by-zero handling*/
	if (fabs(yaw_rate) < 0.00001) {
		yaw_rate = 0.00001;
	}

	/* Predict Next step for each particles*/
	for (int i = 0; i < num_particles; i++) {
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;
		double velocity_div_yaw = velocity / yaw_rate;
		double yaw_delta = yaw_rate * delta_t;

		x = x + velocity_div_yaw * (sin(theta + yaw_delta) - sin(theta));
		y = y + velocity_div_yaw * (cos(theta) - cos(theta + yaw_delta));
		theta = theta + yaw_delta;

		/* Add Noise */
		particles[i].x = x + noise_x(gen);
		particles[i].y = y + noise_y(gen);
		particles[i].theta = theta + noise_theta(gen);
	}

	cout << "Prediction Done " << endl;
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	/* For each observations, find the closest predicted measurements*/
	for (int i = 0; i < observations.size(); i++) {
		double min_distance = FLT_MAX;
		for (int j = 0; j < predicted.size(); j++) {
			double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
			if (distance < min_distance) {
				min_distance = distance;
				observations[i].id = j;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	cout << "Update Weight Start " << endl;

	for (unsigned int i = 0; i < particles.size(); i++) {
		Particle dp = particles[i];

		std::vector<LandmarkObs> map_observations;
		for (unsigned int j = 0; j < observations.size(); j++) {
			LandmarkObs d_observation = observations[j];
			LandmarkObs map_observation = LandmarkObs();

			map_observation.x = dp.x + (cos(dp.theta) * d_observation.x) - (sin(dp.theta) * d_observation.y);
			map_observation.y = dp.y + (sin(dp.theta) * d_observation.x) + (cos(dp.theta) * d_observation.y);

			map_observations.push_back(map_observation);
		}

		std::vector<LandmarkObs> predicted;
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			float landmark_x = map_landmarks.landmark_list[j].x_f;
			float landmark_y = map_landmarks.landmark_list[j].y_f;
			int landmark_id = map_landmarks.landmark_list[j].id_i;
			double distant = dist(dp.x, dp.y, landmark_x, landmark_y);
			if (distant < sensor_range) {
				LandmarkObs temp = LandmarkObs();
				temp.id = landmark_id;
				temp.x = landmark_x;
				temp.y = landmark_y;
				predicted.push_back(temp);
			}
		}

		cout << "Weight " << i << " -  Landmark in sensor range: "  << predicted.size() <<endl;

		dataAssociation(predicted, map_observations);

		particles[i].associations.clear();
		particles[i].sense_x.clear();
		particles[i].sense_y.clear();
		double sig_x = std_landmark[0];
		double sig_y = std_landmark[1];
		double weight = 1.0;

		for (unsigned int j = 0; j < map_observations.size(); j++) {
			LandmarkObs d_observation = map_observations[j];
			double x_obs = d_observation.x;
			double y_obs = d_observation.y;

			LandmarkObs associated = predicted[d_observation.id];
			particles[i].associations.push_back(associated.id);
			particles[i].sense_x.push_back(associated.x);
			particles[i].sense_y.push_back(associated.y);

			double gauss_norm = 1.0 / (2 * M_PI * sig_x * sig_y);
			double exponent = pow((x_obs - associated.x), 2.0) / (2.0 * pow(sig_x, 2.0)) + pow((y_obs - associated.y), 2.0) / (2.0 * pow(sig_y, 2.0));
			double new_weight = (gauss_norm * exp(-1 * exponent));
			// cout << "    exponent " << j << ":" << exponent << endl;
			// cout << "    x - diff " << j << ":" << x_obs - associated.x << endl;
			// cout << "    y - diff " << j << ":" << y_obs - associated.y << endl;
			// cout << "  new_weight " << j << ":" << new_weight << endl;
			weight *= new_weight;
		}

		particles[i].weight = weight;
		weights[i] = weight;
		// cout << "Update Weight " << i << ":" << weight <<endl;
	}
	cout << "Update Weight Done. " << endl;
}

/*** Resample particles with replacement with probability proportional to their weight. ***/
void ParticleFilter::resample() {

	cout << "Resample Start. " << endl;
	/* Normalize the Weight */
	double sum_weight = 0.0;
	for (int i = 0; i < num_particles; i++) {
		sum_weight += particles[i].weight;
	}
	for (int i = 0; i < num_particles; i++) {
		weights[i] = particles[i].weight / sum_weight;
	}

	/* Create discrete distribution for resampling*/
	default_random_engine gen;
	std::discrete_distribution<unsigned int> dist(weights.begin(), weights.end());

	std::vector<Particle> sampled_particles;
	for (int i = 0; i < num_particles; i++) {
		Particle newOne = particles[dist(gen)];
		sampled_particles.push_back(newOne);
	}

	particles = sampled_particles;

	cout << "Resample done. " << endl;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
