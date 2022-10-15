clear // LIMPA ESPAÇO DE VARIÁVEIS
clc // LIMPA CONSOLE DE COMANDOS
xdel(winsid()) // FECHA TODAS AS JANELAS / FIGURAS

//ESTE ARQUIVO É BASEADO NA FUNÇÃO ann_FFBP_lm

// ANN FeedForward Backpropagation Levenberg–Marquardt algorithm training function

// W = ann_FFBP_lm(P,T,N,af,mu,mumax,theta,itermax,mse_min,gd_min)
//
//    Copyright 2011 Trity Technologies.
//    
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 2 of the License, or
//    (at your option) any later version.
//    
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//    
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/
//
// Calling Sequence
//     W = ann_FFBP_lm(P,T,N)
//     W = ann_FFBP_lm(P,T,N,af,mu,mumax,theta,itermax,mse_min,gd_min)
//
// Parameters
//     P : Training input
//     T : Training target
//     N : Number of Neurons in each layer, incluing Input and output layer
//     af : Activation Function from 1st hidden layer to the output layer
//     mu : lm mu value
//     mumax : Maximum allowed mu value
//     theta : mu mutiplier
//     itermax : Maximum iteration
//     mse_min : Minumum Error (Performance Goal)
//     gd_min : Minimum Gradient
//     W : Output Weight and bias
//
// Description
//    This function perform FeedForward Backpropagation with Levenberg–Marquardt algorithm training function.
//    
// 
// See also
//     ann_FFBP_gd
//     ann_FFBP_gda
//     ann_FFBP_gdm
//     ann_FFBP_gdx
//     ann_FFBP_run
//
// Authors
//     Tan C.L. 



// ESPAÇO PARA DECLARAÇÃO DO DADOS DE TREINAMENTO E USO DA REDE NEURAL
// P = MATRIZ DE DADOS DE ENTRADA: LINHAS REPRESENTAM AS DIFERENTES VARIÁVEIS; COLUNAS SÃO AS DIFERENTES AMOSTRAS (PADRÕES)
// T = MATRIZ DE DADOS DE SAÍDA: LINHAS REPRESENTAM AS DIFERENTES VARIÁVEIS DE SAÍDA; COLUNAS SÃO AS DIFERENTES AMOSTRAS (PADRÕES)
// QUANTIDADES DE COLUNAS DE P E T DEVEM COINCIDIR
P = [5.00	6.00	7.00	8.00	9.00	5.00	6.00	7.00	8.00	9.00	5.00	6.00	7.00	8.00	9.00	5.00	6.00	7.00	8.00	9.00	5.00	6.00	7.00	8.00	9.00; 5.00	5.00	5.00	5.00	5.00	6.00	6.00	6.00	6.00	6.00	7.00	7.00	7.00	7.00	7.00	8.00	8.00	8.00	8.00	8.00	9.00	9.00	9.00	9.00	9.00];
T = [25.00	30.00	35.00	40.00	45.00	30.00	36.00	42.00	48.00	54.00	35.00	42.00	49.00	56.00	63.00	40.00	48.00	56.00	64.00	72.00	45.00	54.00	63.00	72.00	81.00];

P_ = P;
T_ = T;


//N É O VETOR QUE DEFINE A TOPOLOGIA DA RNA; CADA POSIÇÃO DO VETOR É REFERENTE À QUANTIDADE DE NEURÔNIOS POR CAMADA DA RNA
// O PRIMEIRO VALOR COINCIDE COM A QUANTIDADE DE ENTRADAS DA REDE NEURAL (FALSOS NEURÔNIOS)

N = [2 2 1];

// af É O VETOR QUE DEFINE A FUNÇÃO DE ATIVAÇÃO DOS NEURÔNIOS DE CADA CAMADA DA REDE - PARA CAMADA DE ENTRADA NÃO É DEFINIDA FUNÇÃO DE ATIVAÇÃO ("FALSO NEURÔNIO")
// ESSAS SÃO AS FUNÇÕES DE ATIVAÇÃO DISPONÍVEIS: 'ann_tansig_activ' - 'ann_purelin_activ' - 'ann_hardlim_activ' - 'ann_logsig_activ'

// NO EXEMPLO ABAIXO A FUNÇÃO DE ATIVAÇÃO 'ann_tansig_activ' É REFERENTE A SEGUNDA CAMADA (CAMADA ESCONDIDA - HIDDEN LAYER) E A FUNÇÃO DE ATIVAÇÃO 'ann_purelin_activ' É REFERENTE A CAMADA DE SAÍDA

af = ['ann_tansig_activ','ann_purelin_activ'];


//--------REAMOSTRAGEM------
// CÓDIGO PARA REAMOSTRAGEM COM REPOSIÇÃO DO CONJUNTO DE TREINAMENTO 

PT = [P_;T_];
s = sample(5,PT,'c'); // 1º ARGUMENTO: QUANTIDADE DE AMOSTRAS; 2º ARGUMENTO: MATRIZ PARA AMOSTRAGEM; 3º ARGUMENTO: AMOSTRAGEM POR COLUNA
P = s(1:2,:);
T = s(3,:);


//---------------------------------------------

// DEFINIÇÃO DE PARÂMETROS DE TREINAMENTO 
// PARTES DO CÓDIDO FORAM COMENTADAS POIS ERAM RELACIONADAS À FUNÇÃO ann_FFBP_lm


        mu = 0.001; // TAXA DE APRENDIZADO

        mumax = 1000//00000; // TAXA DE APRENDIZADO MÁXIMA

        theta = 10;     // FATOR DE AJUSTE DA TAXA DE APRENDIZADO

        itermax = 300; // NÚMERO MÁXIMO DE ITERAÇÕES PARA PARADA DO TREINAMENTO

        mse_min = 1e-5; // MSE ALVO PARA PARADA DO TREINAMENTO

        gd_min =  1e-5; // GRADIENTE MÍNIMO PARA CONTINUIDADE DO TREINAMENTO


hist_mse = [];  // DEFINIÇÃO DO VETOR DE HISTÓRICO DO MSE (MEAN SQUARED ERROR)

//-----INÍCIO DO PROCESSO DE TREINAMENTO DA RNA--------

    // Initialization
    format(8);warning('off');
    W = ann_ffbp_init(N,[-1 1]);
    itercnt = 0;
    af_d = strsubst(af,'ann_','ann_d_');
    mse = %inf;
    gd = %inf;
    A = ann_ffbp_init(N,[0 0]);
    tempW = A;
    train_N = size(P,2);

    // Initialize Training Progress GUI
    handles = ann_training_process();
    handles.itermax.string = string(itermax);
    handles.msemin.string = string(mse_min);
    handles.gdmax.string = 'inf';
    handles.gdmin.string = string(gd_min);
layers = size(N,2)-1; // layers here counted from 1st hidden layers to output layer
n = list(0);
a = list(0);
m = list(0);
s = list(0);
  
   
  
  
  // ------- INÍCIO: WHILE LOOP PARA CONTROLE DO PROCESSO TREINAMENTO ---------
      while mse > mse_min & itercnt < itermax & mu <= mumax & gd > gd_min
        mucnt = 0;
        // Simulate Phase
        n(1) = W(1)(:,1:$-1)*P + repmat(W(1)(:,$),1,size(P,2)); // This could be save in temp n to save memory
        a(1) = evstr(af(1)+'(n('+string(1)+'))');
        for cnt = 2:layers
            n(cnt) = W(cnt)(:,1:$-1)*a(cnt-1) + repmat(W(cnt)(:,$),1,size(P,2)); // This could be save in temp n to save memory
            a(cnt) = evstr(af(cnt)+'(n('+string(cnt)+'))');
        end

// -----CÁLCULO DO ERRO---- SAÍDA ESPERADA - SAÍDA DA REDE

        e = T - a($);

        
        [r,c] = size(a(layers));
        
        
        m(layers) = evstr(af_d(layers)+'(a('+string(layers)+'))'); 
        s(layers) = -(m(layers).*.ones(1,r)).*(ones(1,c).*.eye(r,r));      
    for cnt = layers-1:-1:1     
        Wpre = W(cnt+1)(:,1:$-1);        
        a(cnt) = a(cnt).*.ones(1,N($));
        m(cnt) = evstr(af_d(cnt)+'(a('+string(cnt)+'))');
        s(cnt) = m(cnt).*(Wpre'*s(cnt+1));
    end
       
        Jj = [];
    
        jac = ann_calcjac(kron(P,ones(1,N($))),s(1));
        Jj = [Jj jac s(1)'];
        for cnt = 2:layers
            jac = ann_calcjac(a(cnt-1),s(cnt));
            Jj=[Jj jac s(cnt)'];
        end
        
        mse = (mean(e.^2))
        mse2 = %inf;
        J = Jj;
        J2 = (J' * J);          
        Je = J'*e(:);
        // Calculate Jacobian Matrix
        while  mse2 >= mse & mu <= mumax //round(10e10*mse2)/10e10 >= round(10e10*mse)/10e10         
            dx = -(J2 + (eye(J2)*mu)) \ (Je);
            szpre = 0;
            for cnt = 1:layers
                sz = N(cnt)*N(cnt+1) + N(cnt+1);
                dx_part = dx(szpre+1:szpre+sz);
                A(cnt) = [matrix(dx_part(1:$-N(cnt+1)),N(cnt+1),N(cnt)) dx_part($-N(cnt+1)+1:$)];
                tempW(cnt) = W(cnt) + A(cnt);
                szpre = szpre + sz;
            end
           
           
            // Simulate Phase
            y = ann_FFBP_run(P,tempW,af);
            e2 = T - y;
            mse2 = (mean(e2.^2));
            if  mse2 >= mse //round(10e10*mse2)/10e10 >= round(10e10*mse)/10e10
                mu = mu*theta;
            end

        end

   W = tempW;  //----MATRIZ DE PESOS É ATUALIZADA AQUI------


        mu = mu/theta;
        if (mu < 1e-20)   
             mu = 1e-20;
                    // break
        end

        // Stopping Criteria

        mse = mean(e.^2);

        itercnt = itercnt + 1;
         
        gd = 2*sqrt(Je'*Je)/train_N;

       // Display Training Progress GUI
       if  itercnt == 1 then
           mse_max = mse;
           handles.msemax.string = string(mse_max);
           gd_max = gd;
           handles.gdmax.string = string(gd_max);           
           mse_span = log(mse) - log(mse_min);
           iter_span = itermax;
           gd_span =  log(gd) - log(gd_min);
       end

    
   
       
        // Scilab 5.5 above
        handles.iter.value = round((itercnt/iter_span)*100);
        handles.mse.value = -(log(mse)-log(mse_max))/mse_span * 100;// round(((log(mse) - log(mse_min))/mse_span)*100);
        handles.gd.value = -(log(gd)-log(gd_max))/gd_span * 100; //round(((log(gd) - log(gd_min))/gd_span)*100);

        handles.itercurrent.string = string(itercnt);
        handles.msecurrent.string = string(mse);     
        handles.gdcurrent.string = string(gd);      


// ---- GERAÇÃO DO HISTÓRICO DO MSE DURANTE O TREINAMENTO
hist_mse(itercnt+1)= mse;   


    end
// ----------------------- FIM: WHILE LOOP PARA CONTROLE DO PROCESSO TREINAMENTO------------


//-------HISTÓRICO DO ERRO DURANTE TREINAMENTO -------
figure(100);
//plot(hist_mse);
plot2d(hist_mse,logflag="nl");
//-------- VISUALIZAÇÃO SAÍDAS DA REDE E PADRÕES DE TREINAMENTO -----
y = ann_FFBP_run(P,W,af);
figure(101);
plot(y,'.b');
plot(T,'+r');
legend('Saída da Rede', 'Saída Esperada')
//--------- VISUALIZAÇÃO DA DIFERENÇA ENTRE SAÍDAS DA REDE E PADRÕES DE TREINAMENTO
err = T-y;
figure(102);
plot(err,'.b');


// TESTE DA REDE PARA ANÁLISE DA EXTRAPOLAÇÃO
teste = ann_FFBP_run([0.1:0.1:20; 0.1:0.1:20],W,af);
testex = [0.1:0.1:20].*[0.1:0.1:20];
figure(103);
plot(testex,teste);
plot(testex,testex,'r')
legend('Saída da Rede', 'Saída Esperada')


