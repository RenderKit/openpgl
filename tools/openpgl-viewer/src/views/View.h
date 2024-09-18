#pragma once

// #include "../data/Data.h"
struct Data;

struct View
{
    View(Data *data);
    virtual ~View(){};
    virtual void dataUpdated() = 0;
    virtual void drawUI() = 0;

    virtual void draw() {};
    virtual void drawViewport() {};

   protected:
    Data *m_data;
};