#ifndef PROSWARM_H_
#define PROSWARM_H_

// STANDARD
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <random>
#include <limits>

// ROOT
#include "TFile.h"
#include "TCanvas.h"
#include "TH1D.h"
#include "THStack.h"
#include "TLegend.h"

// EIGEN
#include <Eigen/Dense>
#include <Eigen/SVD>

// PROfit
#include "PROconfig.h"
#include "PROserial.h"
#include "PROmetric.h"
#include "PROgress.h"

namespace PROfit{

    class PROswarm {
        public:
            PROswarm(PROmetric &metric, std::mt19937 &gen, const std::vector<std::vector<float>> &latin_starts, const Eigen::VectorXf& ilb, const Eigen::VectorXf& iub ,size_t max_iterations ) : max_iterations(max_iterations), lb(ilb),ub(iub) {

                num_particles = latin_starts.size();
                npar = lb.size();

                global_best_chi = std::numeric_limits<float>::max();
                global_best_position = Eigen::VectorXf(npar);

                particles.resize(num_particles, Eigen::VectorXf(npar));
                velocities.resize(num_particles, Eigen::VectorXf(npar));
                personal_best_positions.resize(num_particles, Eigen::VectorXf(npar));
                personal_best_chis.resize(num_particles, std::numeric_limits<float>::max());
                grad.resize(npar);
    
                fixed.resize(npar);

                for (size_t j = 0; j < npar; ++j) {
                    if( std::isinf(lb(j))) lb(j)=-3;
                }

                for (size_t i = 0; i < num_particles; ++i) {
                    for (size_t j = 0; j < npar; ++j) {
                        particles[i](j) = latin_starts[i][j];
                        velocities[i](j) = ub(j)==lb(j) ? 0.0 : randomFloat(gen,-0.1 * (ub(j) - lb(j)), 0.1 * (ub(j) - lb(j)));
                        fixed[j] = ub(j)==lb(j) ? 1 : 0 ;
                    }

                    // Evaluate initial position
                    float chi = metric(particles[i],grad,false);
                    personal_best_chis[i] = chi;
                    personal_best_positions[i] = particles[i];
                    if (chi < global_best_chi) {
                        global_best_chi = chi;
                        global_best_position = particles[i];
                    }
                }//
            }

            void runSwarm(PROmetric &metric,std::mt19937 &gen, MultiPROgressBar * pin = NULL) {
                const float inertia_w_start = 0.9; 
                const float inertia_w_end = 0.6; 
                const float cognitive_w = 2.0; 
                const float social_w = 2.0; 
                const size_t max_stagnant_iterations =50;
                const float  convergence_threshold = 1e-4;
                size_t stagnant_iterations = 0;
                float previous_best_chi = global_best_chi;

                const float vmax_factor = 0.1;  
                Eigen::VectorXf vmax = vmax_factor * (ub - lb);

                log<LOG_INFO>(L"%1% ||  Swarm starting with %2% inertia, %3% cognitive and %4% social weights  ") % __func__ % inertia_w_start % cognitive_w % social_w  ;


                for (size_t iter = 0; iter < max_iterations; iter++) {
                        if(pin!=NULL)pin->increment_bar(1);
                        //update inertia
                        float inertia_w = inertia_w_start - (static_cast<float>(iter) / max_iterations) * (inertia_w_start - inertia_w_end);

                    for (size_t i = 0; i < num_particles; i++) {
                        // Update velocity
                        for (size_t j = 0; j < npar; ++j) {
                            float r1 = randomFloat(gen,0.0, 1.0);
                            float r2 = randomFloat(gen,0.0, 1.0);
                            if(fixed(j)){
                                velocities[i](j)=0.0;
                            }else{
                                velocities[i](j) = inertia_w * velocities[i](j) + cognitive_w * r1 * (personal_best_positions[i](j) - particles[i](j)) + social_w * r2 * (global_best_position(j) - particles[i](j));
                            //clamp worth it worth it??
                            velocities[i](j) = std::max(-vmax(j), std::min(vmax(j), velocities[i](j)));
                            }
                          //if(std::isinf(velocities[i][j])){
                          //      log<LOG_INFO>(L"%1% || INF %2%, r1 %3% r2 %4% par %5% vmax %6% ub %7% lb %8% particle %9% PB %10% GB %11% ") % __func__ % velocities[i](j) % r1 % r2 %j % vmax(j) % ub(j) % lb(j) % particles[i](j) % personal_best_positions[i](j) % global_best_position(j)   ;
                          //     exit(EXIT_FAILURE); 

                          // }
                        }

                        // Update position
                        particles[i] += velocities[i];

                        // Apply bounds and  damp
                        for (size_t j = 0; j < npar; ++j) {
                             if (particles[i](j) <= lb(j) || particles[i](j) >= ub(j)) {
                                 velocities[i](j) *= -0.5;  
                                 particles[i](j) = std::max(lb(j), std::min(ub(j), particles[i](j)));
                            }
                        }


                        float chi = metric(particles[i],grad,false);
                        function_calls++;

                        


                        if (chi < personal_best_chis[i]) {
                            personal_best_chis[i] = chi;
                            personal_best_positions[i] = particles[i];
                        }
                        if (chi < global_best_chi) {
                            global_best_chi = chi;
                            global_best_position = particles[i];
                        }



                    }

                    // Check for stagnation
                        if (std::abs(global_best_chi - previous_best_chi) < convergence_threshold) {
                            stagnant_iterations++;
                        } else {
                            stagnant_iterations = 0;
                        }
                        previous_best_chi = global_best_chi;

                        // Early stopping condition
                        if (stagnant_iterations >= max_stagnant_iterations) {
                            log<LOG_INFO>(L"%1% ||  Early Stopping of swarm at %2% due to stagnation ") % __func__ % iter  ;
                            break;
                        }
                    //if(iter%50==0)log<LOG_INFO>(L"%1% ||  Swarm at iter %2% has particle got %3% at %4% ") % __func__ % iter % global_best_chi % global_best_position ;

                }
                log<LOG_INFO>(L"%1% ||  Swarm ran for %2% and the best particle got %3% at %4% ") % __func__ % max_iterations % global_best_chi % global_best_position ;

            }

            Eigen::VectorXf getGlobalBestPosition() const {
                return global_best_position;
            }

            float getGlobalBestScore() const {
                return global_best_chi;
            }

            size_t getFunctionCalls() const {
                return function_calls;
            }
        private:

            size_t num_particles;
            size_t npar;
            size_t max_iterations;
            Eigen::VectorXf lb;
            Eigen::VectorXf ub;
            Eigen::VectorXf grad;
            Eigen::VectorXf fixed;
            std::vector<Eigen::VectorXf> particles;
            std::vector<Eigen::VectorXf> velocities;
            std::vector<Eigen::VectorXf> personal_best_positions;
            std::vector<float> personal_best_chis;
            Eigen::VectorXf global_best_position;
            float global_best_chi;
            size_t function_calls = 0;

            float randomFloat(std::mt19937 &gen, float min, float max) {
                std::uniform_real_distribution<float> dist(min, max);
                return dist(gen);
            }
    };

}


#endif
