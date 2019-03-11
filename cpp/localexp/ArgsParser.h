#pragma once
#include <vector>
#include <map>
#include <string>

namespace specialized_func {
	template <typename T> T convertStringToValue(std::string str)
	{
		return (T) std::stod(str);
	}
	template <> float convertStringToValue(std::string str) { return std::stof(str); }
	template <> int convertStringToValue(std::string str) { return std::stoi(str); }
	template <> std::string convertStringToValue(std::string str) { return str; }
	template <> bool convertStringToValue(std::string str)
	{
		if (str == "true") return true;
		if (str == "false") return false;
		return convertStringToValue<int>(str) != 0;
	}

}


class ArgsParser
{
	std::vector<std::string> args;
	std::map<std::string, std::string> argMap;

	void parseArgments()
	{
		for (int i = 0; i < args.size(); i++)
		{
			if (args[i][0] == '-')
			{
				std::string name(&args[i][1]);
				if (i + 1 < args.size())
				{
					argMap[name] = args[i + 1];
					i++;
					//std::cout << name << ": " << argMap[name] << std::endl;
				}
			}
		}
	}
  
/*	
	template <typename T>
	T convertStringToValue(std::string str) const
	{
		return (T)std::stod(str);
	}
	template <> float convertStringToValue(std::string str)<float> const{ return std::stof(str); }
	template <> int convertStringToValue(std::string str)<int> const{ return std::stoi(str); }
	template <> std::string convertStringToValue(std::string str)<std::string> const { return str; }
	template <> bool convertStringToValue(std::string str)<bool> const
	{
		if (str == "true") return true;
		if (str == "false") return false;
		return convertStringToValue<int>(str) != 0;
	}
*/

// added by CCJ, due to the compilation error of the previous VS Studio code when compiled in g++;
// error: explicit specialization in non-namespace scope ‘class ArgsParser’
// see https://stackoverflow.com/questions/3052579/explicit-specialization-in-non-namespace-scope;
	template <typename T>
	T convertStringToValue(std::string str) const
	{
		return specialized_func::convertStringToValue<T>(str);
	}



public:
	ArgsParser(){}
	ArgsParser(int argn, const char **args)
	{
		for (int i = 0; i < argn; i++) {
			this->args.push_back(args[i]);
		}
		parseArgments();
	}
	ArgsParser(const std::vector<std::string>& args)
	{
		this->args = args;
		parseArgments();
	}

	template <typename T>
	bool TryGetArgment(std::string argName, T& value) const
	{
		auto it = argMap.find(argName);
		if (it == argMap.end())
			return false;

		value = convertStringToValue<T>(it->second);
		return true;
	}
};
