clear
clc

function [norm_matrix] = norm_img(matrix_img)
    norm_matrix = zeros(28, 28)
    for i = 1 : 28
        for j = 1 : 28
            norm_matrix(i, j) = matrix_img(i, j)
        end
    end
    norm_matrix = norm_matrix / 255
endfunction

function [input_vec, target_vec, dir_list] = get_input_vec(folder, digits, samples)
    error = 0
    input_vec = []
    dir_list = cell(1, samples*2)
    for j = 1 : 2 //1 - 10
        if j == 1 then
            tar = ones(1, samples)
            target_vec = tar
        else
            tar = (-1) * ones(1, samples)
            target_vec = [target_vec tar]
        end
        directory = "mnist_dataset\" + folder + "\" + string(digits(j))
        archives = dir(directory)
        archives_list = archives(2)
        //n = size(archives_list, 1)
        for i = 1 : samples //1 - n
            file_loc = directory + "\" + archives_list(i)
            if j == 1 then
                dir_list{i} = file_loc
            else
                dir_list{i+samples} = file_loc
            end
            img = imread(file_loc)
            try
                grayscale_img = rgb2gray(img)
            catch
                disp("Error")
                error = error + 1
            end
            normalized_vec = norm_img(grayscale_img)
            normalized_vec = normalized_vec(:)
            input_vec = [input_vec normalized_vec]
        end
    end
endfunction

function [y] = signal(u)
    if u >= 0 then
        y = 1
    else
        y = -1
    end
endfunction

function [w] = train_perceptron(digits, samples, learning_rate)
    tic()
    epoch = 0
    error = 1
    
    [x_train, d] =  get_input_vec("training", digits, samples)
    [m, n] = size(x_train)

    x_train = [(-1) * ones(1, n); x_train] // x_train com bias -1
    w = 0 + (1 - 0) * rand(m+1, 1) // pesos aleatórios entre 0 e 1
    
    while error == 1
        error = 0
        for j = 1 : n
            u = w' * x_train(:, j)
            y = signal(u)
            if y ~= d(j) then
                w = w + learning_rate * (d(j) - y) * x_train(:, j)
                error = 1
            end
        end
        if error == 0 then
            break
        end
        epoch = epoch + 1
        printf("\n\n========= Época (%d) ===========\n", epoch)
        printf("Pesos sinápticos ajustados.\n")
        printf("===============================\n")
    end

    elapsed_time = toc() // Para o contador e atribui o tempo a variavel elapsed_time
    printf("\nFase de treinamento concluída.")
    printf("\nForam necessárias %d épocas para o ajuste necessário.\n", epoch)
    printf("Tempo decorrido: %.2f segundos\n", elapsed_time)
    printf("===============================\n")
endfunction

function test_perceptron(w, digits, samples)
    [x_test, d, img] = get_input_vec("testing", digits, samples)

    [m, n] = size(x_test)
    
    x_test = [(-1) * ones(1, n); x_test]
    errors = 0
    
    printf("\n\nEntradas de teste:")
    
    for i = 1 : n
        u = w' * x_test(:, i)
        y = signal(u)
        if y ~= d(i) then
            errors = errors + 1
        end    
        if d(i) == 1 then
            exp_output = digit1
        end
        if d(i) == -1 then
            exp_output = digit2
        end
        disp(rgb2gray(imread(img{i})))
        if y == 1 then
            printf("Número reconhecido: %d\n", digit1)
        end
        if y == -1 then
            printf("Número reconhecido: %d\n", digit2)
        end
        printf("Saída esperada: %d\n", exp_output)
        printf("==================================\n")
    end
    
    printf("Quantidade de erros: %d\n", errors)
    rate = ((n - errors) / n) * 100 // Calcula a taxa de acertos
    printf("Taxa de acertos: %.2f%%", rate)
endfunction

digit1 = input("Primeiro conjunto de digitos: ")
digit2 = input("Segundo conjunto de digitos: ")
samples = input("Quantidade de amostras: ")
learning_rate = input("Digite a taxa de aprendizagem: ")
digits = [digit1; digit2]

w = train_perceptron(digits, samples, learning_rate)

printf("\nFase de testes:\n")
samples = input("Insira a quantidade de imagens de cada digito a serem analisadas: ")

test_perceptron(w, digits, samples)
