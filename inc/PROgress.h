#ifndef PROGRESS_H_
#define PROGRESS_H_

//C++ include 
#include <Eigen/Eigen>
#include <map>
#include <cmath>
#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>
#include <vector>
#include <iomanip>
#include <string>

// Our include
#include "PROconfig.h"
#include "PROcreate.h"
#include "PROlog.h"

namespace PROfit {

    class PROgressBar {
        private:
            int current;
            int total;
            int bar_width;
            std::string description;
            mutable std::mutex print_mutex;

            void clear_line() const {
                std::cerr << "\r" << std::string(80, ' ') << "\r";
            }

        public:
            PROgressBar(int total, int bar_width = 30, const std::string& desc = "Progress")
                : current(0), total(total), bar_width(bar_width), description(desc) {}

            void update(int value = -1) {
                std::lock_guard<std::mutex> lock(print_mutex);
                if (value == -1) {
                    current++;
                } else {
                    current = value;
                }
                display_locked();
            }

            void increment() {
                std::lock_guard<std::mutex> lock(print_mutex);
                current++;
                display_locked();
            }

            void set_progress(int value) {
                std::lock_guard<std::mutex> lock(print_mutex);
                current = value;
                display_locked();
            }

            void display() const {
                std::lock_guard<std::mutex> lock(print_mutex);
                display_locked();
            }

        private:
            void display_locked() const {
                int curr = current;
                float progress = static_cast<float>(curr) / total;
                int pos = static_cast<int>(bar_width * progress);

                clear_line();
                std::cerr << std::left << std::setw(15) << description << " [";

                for (int i = 0; i < bar_width; ++i) {
                    if (i < pos) std::cerr << "█";
                    else if (i == pos) std::cerr << ">";
                    else std::cerr << " ";
                }

                std::cerr << "] " << std::fixed << std::setprecision(1) 
                    << std::setw(5) << (progress * 100.0) << "% "
                    << "(" << std::setw(3) << curr << "/" << std::setw(3) << total << ")";
                std::cerr.flush();
            }

        public:
            void finish() {
                std::lock_guard<std::mutex> lock(print_mutex);
                current = total;
                display_locked();
            }

            int get_current() const {
                std::lock_guard<std::mutex> lock(print_mutex);
                return current;
            }

            bool is_complete() const {
                std::lock_guard<std::mutex> lock(print_mutex);
                return current >= total;
            }

            int get_total() const {
                return total; 
            }

            PROgressBar(PROgressBar&& other) noexcept 
                : current(other.current), total(other.total), 
                bar_width(other.bar_width), description(std::move(other.description)) {}

            PROgressBar& operator=(PROgressBar&& other) noexcept {
                if (this != &other) {
                    std::lock_guard<std::mutex> lock1(print_mutex);
                    std::lock_guard<std::mutex> lock2(other.print_mutex);
                    current = other.current;
                    total = other.total;
                    bar_width = other.bar_width;
                    description = std::move(other.description);
                }
                return *this;
            }

            PROgressBar(const PROgressBar&) = delete;
            PROgressBar& operator=(const PROgressBar&) = delete;
    };

    class MultiPROgressBar {
        private:
            std::vector<PROgressBar> bars;
            mutable std::mutex display_mutex;
            int num_bars;
            std::thread refresh_thread;
            std::atomic<bool> should_stop{false};
            std::atomic<bool> needs_refresh{false};

        public:
            MultiPROgressBar(const std::vector<std::pair<int, std::string>>& bar_configs) {
                num_bars = bar_configs.size();
                bars.reserve(bar_configs.size()); // Reserve space to avoid reallocation


                for (size_t i = 0; i < bar_configs.size(); ++i) {
                    const auto& config = bar_configs[i];
                    // config.first = total increments, config.second = name
                    bars.emplace_back(config.first, 30, config.second);

                }
            }
            void start_display_thread() {
                should_stop = false;
                refresh_thread = std::thread([this]() {
                        while (!should_stop.load() && !all_complete()) {
                        if (needs_refresh.load()) {
                        refresh_display();
                        needs_refresh = false;
                        }
                        std::this_thread::sleep_for(std::chrono::milliseconds(300));
                        }
                        });
            }


            void increment_bar(int bar_index) {
                if (bar_index >= 0 && bar_index < num_bars) {
                    bars[bar_index].increment();
                    needs_refresh = true;
                }
            }

            void finish_all() {
                should_stop = true;
                if (refresh_thread.joinable()) {
                    refresh_thread.join();
                }

                //std::cerr << "\nAll tasks completed!" << std::endl;
                log<LOG_ERROR>(L"   ");

            }

            // Set progress bar to specific value
            void set_bar_progress(int bar_index, int value) {
                if (bar_index >= 0 && bar_index < num_bars) {
                    bars[bar_index].set_progress(value);
                    refresh_display();

                } else {
                }
            }

            void refresh_display() {
                std::lock_guard<std::mutex> lock(display_mutex);
                // Move cursor up to overwrite previous bars
                std::cerr << "\033[" << num_bars << "A";

                for (auto& bar : bars) {
                    bar.display();
                    std::cerr << std::endl;
                }
            }

            void initialize_display() {
                // Create empty lines for the progress bars
                for (int i = 0; i < num_bars; ++i) {
                    std::cerr << std::endl;
                }
                refresh_display();
            }



            bool all_complete() const {
                for (const auto& bar : bars) {
                    if (!bar.is_complete()) {
                        return false;
                    }
                }
                return true;
            }

            int get_bar_progress(int bar_index) const {
                if (bar_index >= 0 && bar_index < num_bars) {
                    return bars[bar_index].get_current();
                }
                return -1;
            }
    };
}

#endif
