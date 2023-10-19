//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//% Classificação binária de dígitos manuscritos  %
//% através de perceptron de camada única         %
//%                                               %
//% Autores: Arthur Hirury Hwang Bo;              %
//%          Dr. Alexandre Maniçoba de Oliveira;  %
//%          Andrés Mendes Tercero;               %
//%          João Pedro Ribeiro Bezerra;          %
//%          Matheus Cardoso Medeiros             %
//%                                               %
//%                                               %
//% 13/09/2023                                    %
//% Feito para Scilab Versão 2023.1.0             %
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear // limpa a memória
clc // limpa o console

// Função para normalizar a matriz da imagem passada como parâmetro
// todos os valores da matriz serão dividos por 255 e retornados ao final
function [norm_matrix] = norm_img(matrix_img)
    norm_matrix = zeros(28, 28)
    for i = 1 : 28
        for j = 1 : 28
            norm_matrix(i, j) = matrix_img(i, j)
        end
    end
    norm_matrix = norm_matrix / 255
endfunction

// Função para gerar as matrizes de entrada do perceptron. Recebe como parâmetro a pasta
// a ser acessada (training ou testing), o conjunto de dígitos, e quantidade de amostras.
// Retorna a matriz de entrada, o vetor contendo as saídas esperadas e a lista de diretórios que foram acessados.
function [input_vec, target_vec, dir_list] = get_input_vec(folder, digits, samples)
    error = 0
    input_vec = []
    dir_list = cell(1, samples*2) // Células para guardar todos os diretórios acessados
    for j = 1 : 2 // Itera 2 vezes sob o vetor digits, afim de acessar as duas pastas em cada iteração
        if j == 1 then // Verifica se o loop está sendo executado pela primeira vez
            tar = ones(1, samples)
            target_vec = tar // popula o vetor target_vec com as saídas esperadas (1)
        else
            tar = (-1) * ones(1, samples)
            target_vec = [target_vec tar] // popula a metade do vetor target_vec com as saídas esperadas (-1)
        end
        directory = "mnist_dataset\" + folder + "\" + string(digits(j)) // Cria string do caminho completo do diretório
        archives = dir(directory)
        archives_list = archives(2) // matriz contendo todos os arquivos que estão contidos no diretório da string directory
        //n = size(archives_list, 1)
        for i = 1 : samples //1 - n Itera sob o número de amostras escolhido
            file_loc = directory + "\" + archives_list(i) // Localização do arquivo completa do índice i
            if j == 1 then // verifica se o primeiro loop foi executado pela primeira vez
                dir_list{i} = file_loc // popula a célula dir_list com a localização completa dos arquivos
            else
                dir_list{i+samples} = file_loc // popula a célula com a localização completa dos arquivos referentes à segunda iteração
            end
            img = imread(file_loc) // Lê a imagem localizada em file_loc, e a armazena como uma matriz na varíavel img
            try
                grayscale_img = rgb2gray(img) // tenta converter a imagem para escala de cinza
            catch
                disp("Error")
                error = error + 1
            end
            normalized_vec = norm_img(grayscale_img) // Executa a função de normalização da matriz e atribui o resultado à variável normalized_vec
            //normalized_vec = normalized_vec(:)
            input_vec = [input_vec normalized_vec(:)] // popula a matriz input_vec com os vetores coluna das imagens normalizadas
        end
    end
endfunction


// Função Sinal (Função de ativação)
// Retorna o valor de saída do perceptron [y]
function [y] = signal(u)
    if u >= 0 then // Verifica se o potencial de ativação é maior ou igual à zero
        y = 1 // caso for, y = 1
    else
        y = -1 // caso contrário, y = -1
    end
endfunction

// Função para ajustes sinápticos do conjunto de treinamento do perceptron.
// Recebe como parâmetros o vetor de dígitos a serem analisado,
// A quantidade de amostras de cada dígito e a taxa de aprendizagem.
// Retorna o vetor w já ajustado
function [w] = train_perceptron(digits, samples, learning_rate)
    tic() // Inicializa o contador de tempo
    epoch = 0 // Inicializa a variável contadora de épocas com 0
    error = 1 // Inicializa error como 1: flag para o loop while.
    
    // Executa a função get_input_vec e atribui às variáveis x_train e d a matriz com as imagens de treinamento e as saídas esperadas, respectivamente.
    [x_train, d] =  get_input_vec("training", digits, samples)
    [m, n] = size(x_train) // atribui às variáveis m e n o número de linhas e colunas da matriz x_train.

    x_train = [(-1) * ones(1, n); x_train] // primeira linha da matriz x_train é populada com -1 (limiar de ativação).
    w = 0 + (1 - 0) * rand(m+1, 1) // inicializa o vetor w com pesos aleatórios entre 0 e 1
    
    while error == 1
        error = 0
        for j = 1 : n
            u = w' * x_train(:, j) // Calcula o potencial de ativação (u), sendo a combinação linear entre w' e x_train.
            y = signal(u) // atribui à variável y o valor obtido na função sinal passando u como parâmetro
            if y ~= d(j) then // verifica se a saída y é diferente
                w = w + learning_rate * (d(j) - y) * x_train(:, j) // incrementa o vetor w de acordo com a regra de Hebb.
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

    elapsed_time = toc() // Para o contador e atribui o tempo decorrido à variável elapsed_time
    printf("\n=============================")
    printf("\nFase de treinamento concluída.")
    printf("\nUm total de %d imagens foram utilizadas para treinamento.", samples*2)
    printf("\nForam necessárias %d épocas para o ajuste necessário.\n", epoch)
    printf("Tempo decorrido: %.2f segundos\n", elapsed_time)
    printf("===============================\n")
endfunction

// Função para realizar a fase de testes.
// Recebe como parâmetros o vetor w já ajustado, o vetor de dígitos e as amostras a serem testadas.
function test_perceptron(w, digits, samples)
    // Executa a função get_input_vec afim de obter as amostras do dígitos da pasta testing e atribui
    // a matriz com as imagens de teste à variável x_test, d com as saídas esperadas e img com células
    // de todos os locais dos arquivos acessados.
    [x_test, d, img] = get_input_vec("testing", digits, samples)

    [m, n] = size(x_test) // atribui às variáveis m e n o número de linhas e colunas da matriz x_test.
    
    x_test = [(-1) * ones(1, n); x_test] // primeira linha da matriz x_test é populada com -1 (limiar de ativação).
    errors = 0 // Variável contadora de erros de classificação
    
    printf("\n\nEntradas de teste:")
    
    for i = 1 : n
        u = w' * x_test(:, i) // Cálculo do potencial de ativação (u) a partir dos dados da coluna i da matriz x_test
        y = signal(u) // Cálculo da saída y
        if y ~= d(i) then // verifica se y é diferente da saída esperada.
            errors = errors + 1 // incrementa o erro
        end    
        if d(i) == 1 then // Verifica se a saída esperada no índice i é igual 1
            exp_output = digit1 // exp_output recebe o valor do dígito 1 (saída esperada = 1)
        end
        if d(i) == -1 then // Verifica se a saída esperada no índice i é igual a -1
            exp_output = digit2 // exp_output recebe o valor do dígito 2 (saída esperada = -1)
        end
        disp(rgb2gray(imread(img{i}))) // Mostra a representação matricial da imagem localizada no índice i da célula img
        if y == 1 then // Verifica se y é igual à 1
            printf("Número reconhecido: %d\n", digit1) // Mostra na tela o dígito 1 reconhecido (relacionado à saída esperada 1)
        end
        if y == -1 then // Verifica se y é igual à -1
            printf("Número reconhecido: %d\n", digit2) // Mostra na tela o dígito 2 reconhecido (relacionado à saída esperada -1)
        end
        printf("Saída esperada: %d\n", exp_output) // Mostra qual deveria ser a saída esperada do dígito analisado.
        printf("==================================\n")
    end
    
    printf("Quantidade de imagens analisadas: %d\n", samples*2)
    printf("Quantidade de erros: %d\n", errors)
    rate = ((n - errors) / n) * 100 // Calcula a taxa de acertos
    printf("Taxa de acertos: %.2f%%", rate)
endfunction

printf("======= Fase de treinamento =======\n\n")

digit1 = input("Primeiro conjunto de digitos: ")
digit2 = input("Segundo conjunto de digitos: ")
samples = input("Quantidade de amostras: ")
learning_rate = input("Digite a taxa de aprendizagem: ")
digits = [digit1; digit2]

w = train_perceptron(digits, samples, learning_rate) // Ajusta o vetor w.

printf("\nFase de testes:\n")
samples = input("Insira a quantidade de imagens de cada digito a serem analisadas: ")

test_perceptron(w, digits, samples) // Testa o perceptron com o vetor w ajustado.
