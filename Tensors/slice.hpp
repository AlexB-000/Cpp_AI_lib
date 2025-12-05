#pragma once

class slice{
public:
    slice(unsigned int inStart=0, unsigned int inEnd=-1, int inStep=1) : start(inStart), end(inEnd), step(inStep) {}
private:
    unsigned int start;
    unsigned int end;
    int step;
};