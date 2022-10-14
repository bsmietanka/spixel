/*
    This file is part of spixel.

    Spixel is free software : you can redistribute it and / or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Foobar is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Foobar. If not, see <http://www.gnu.org/licenses/>.
*/


#pragma once

#include <vector>
#include <unordered_set>
#include <mutex>


template <typename T> class Deque {
private:
    std::vector<T> vector;
    size_t start;
    size_t end;
    size_t listSize;
public:
    Deque(size_t maxSize) :
        start(0), end(0), listSize(0)
    {
        vector.resize(maxSize);
    }

    size_t Size() const { return listSize; }

    bool Empty() const { return listSize == 0; }

    void Clear() { start = end = listSize = 0; }

    void PushBack(const T& value)
    {
        if ((end + 1) % vector.size() == start) resize(2 * vector.size());
        vector[end] = value;
        end = (end + 1) % vector.size();
        listSize++;
    }

    const T& Front() const { return vector[start]; }

    T PopFront() 
    { 
        size_t retIndex = start;
        start = (start + 1) % vector.size(); 
        listSize--; 
        return vector[retIndex];
    }

private:
    void resize(size_t newSize)
    {
        std::vector<T> newVector;

        assert(newSize > vector.size());
        newVector.resize(newSize);
        if (start <= end) std::copy(vector.begin() + start, vector.begin() + end, newVector.begin());
        else {
            std::copy(vector.begin() + start, vector.end(), newVector.begin());
            std::copy(vector.begin(), vector.begin() + end, newVector.begin() + vector.size() - start);
        }
        std::swap(vector, newVector);
        start = 0;
        end = listSize;
    }
};
