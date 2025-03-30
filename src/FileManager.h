#ifndef FILE_MANAGER_H
#define FILE_MANAGER_H

#pragma once

#include <filesystem>
#include <string>

class FileManager {
public:
    static inline const std::string DATA_DIR = "../data";
    static inline const std::string DATA_EXT = ".json";
    
    static std::string GetFullPath(const std::string& filename) {
        std::filesystem::path file_path(filename);
        
        // Jeśli podano pełną ścieżkę, użyj jej
        if(file_path.is_absolute()) {
            return filename;
        }
        
        // Dodaj rozszerzenie jeśli brak
        if(file_path.extension().empty()) {
            file_path.replace_extension(DATA_EXT);
        }
        
        // Sprawdź czy plik istnieje w podanej lokalizacji
        if(std::filesystem::exists(file_path)) {
            return file_path.string();
        }
        
        // Sprawdź w DATA_DIR
        std::filesystem::path data_path = DATA_DIR / file_path;
        if(std::filesystem::exists(data_path)) {
            return data_path.string();
        }
        
        // Sprawdź w ../DATA_DIR (dla build/)
        data_path = std::filesystem::path("..") / DATA_DIR / file_path;
        if(std::filesystem::exists(data_path)) {
            return data_path.string();
        }
        
        // Jeśli nie znaleziono, zwróć ścieżkę w DATA_DIR
        return (DATA_DIR / file_path).string();
    }
    
    static void EnsureDataDirExists() {
        if(!std::filesystem::exists(DATA_DIR)) {
            std::filesystem::create_directory(DATA_DIR);
        }
    }
};


#endif // FILE_MANAGER_H