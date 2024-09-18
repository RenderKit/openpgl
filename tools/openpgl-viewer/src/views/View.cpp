
#include "View.h"

#include "../data/Data.h"
View::View(Data *data)
{
    m_data = data;
    m_data->registerView(this);
};