#include <iostream>
#include <string>
#include <thread>
#include <vector>

void Step1(int &v) {
    v += 1;
    std::this_thread::sleep_for(std::chrono::seconds(1));
}

void Step2(int &v) {
    v -= 1;
}

void Callback(int &v) { std::cout << v << std::endl; }

void Workflow(int &input, void (*pfunc)(int &v)) {
    Step1(input);
    Step2(input);
    pfunc(input);
}
// run1 : step1 step2 callback
//  run2: step1 step2 callback

int main() {
    std::thread run([&] {
        for (int i = 0; i < 10; i++) {
            Workflow(i, Callback);
            // std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    });
    std::thread run2([&] {
        for (int i = 0; i < 10; i++) {
            Workflow(i, Callback);
            // std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    });
    run.join();
    printf("This is a test.\n");
    return 0;
}
​​​
