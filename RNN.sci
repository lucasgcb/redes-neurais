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
P = [2.66	1.98	1.44	0.71	0.46	0.05	0.00	0.00	0.00	0.01	0.01	0.03	0.03	0.03	0.06	0.55	1.80	2.83	3.14	3.86	4.25	4.65	4.99	5.30	5.46	5.67	5.79	5.85	6.03	6.21	6.34	6.40	6.45	6.52	6.57	6.60	6.62	6.64	6.68	6.68	6.68	6.68	6.68	6.68	6.67	6.66	6.66	6.66	6.66	6.66	6.66	6.66	6.66	6.66	6.66	6.66	6.66	6.66	6.66	6.66	6.66	6.66	6.66	6.66	6.66	6.66	6.66; 4.44	4.43	4.43	4.43	4.41	4.37	4.27	4.20	4.12	4.05	3.96	3.87	3.77	3.68	3.53	3.43	3.26	3.04	2.93	2.68	2.48	2.20	1.85	1.41	1.11	0.64	0.23	0.11	0.10	0.08	0.08	0.07	0.07	0.07	0.08	0.09	0.09	0.11	0.19	0.29	0.42	0.61	0.79	1.13	1.34	1.55	1.81	2.06	2.26	2.40	2.55	2.74	2.96	3.08	3.24	3.35	3.53	3.71	3.82	3.91	4.06	4.14	4.20	4.28	4.37	4.42	4.45];

T = [0.00	0.06	0.11	0.18	0.20	0.26	0.32	0.36	0.40	0.43	0.48	0.52	0.57	0.61	0.67	0.70	0.75	0.81	0.83	0.88	0.91	0.96	1.01	1.06	1.09	1.15	1.20	1.22	1.27	1.35	1.43	1.47	1.50	1.56	1.61	1.64	1.67	1.70	1.73	1.77	1.79	1.81	1.83	1.89	1.92	1.96	1.99	2.04	2.09	2.12	2.15	2.20	2.26	2.29	2.33	2.36	2.41	2.46	2.50	2.53	2.59	2.62	2.64	2.68	2.73	2.78	2.83];

//N É O VETOR QUE DEFINE A TOPOLOGIA DA RNA; CADA POSIÇÃO DO VETOR É REFERENTE À QUANTIDADE DE NEURÔNIOS POR CAMADA DA RNA
// O PRIMEIRO VALOR COINCIDE COM A QUANTIDADE DE ENTRADAS DA REDE NEURAL (FALSOS NEURÔNIOS)
P_ = P;
T_ = T;

N = [2 5 5 1];

// af É O VETOR QUE DEFINE A FUNÇÃO DE ATIVAÇÃO DOS NEURÔNIOS DE CADA CAMADA DA REDE - PARA CAMADA DE ENTRADA NÃO É DEFINIDA FUNÇÃO DE ATIVAÇÃO ("FALSO NEURÔNIO")
// ESSAS SÃO AS FUNÇÕES DE ATIVAÇÃO DISPONÍVEIS: 'ann_tansig_activ' - 'ann_purelin_activ' - 'ann_hardlim_activ' - 'ann_logsig_activ'

// NO EXEMPLO ABAIXO A FUNÇÃO DE ATIVAÇÃO 'ann_tansig_activ' É REFERENTE A SEGUNDA CAMADA (CAMADA ESCONDIDA - HIDDEN LAYER) E A FUNÇÃO DE ATIVAÇÃO 'ann_purelin_activ' É REFERENTE A CAMADA DE SAÍDA

af = ['ann_purelin_activ','ann_tansig_activ', 'ann_purelin_activ'];


//--------REAMOSTRAGEM------
// CÓDIGO PARA REAMOSTRAGEM COM REPOSIÇÃO DO CONJUNTO DE TREINAMENTO 

PT = [P_;T_];
s = sample(40,PT,'c'); // 1º ARGUMENTO: QUANTIDADE DE AMOSTRAS; 2º ARGUMENTO: MATRIZ PARA AMOSTRAGEM; 3º ARGUMENTO: AMOSTRAGEM POR COLUNA
P = s(1:2,:);
T = s(3,:);
conjunto_teste = [];
for r=1:max(size(PT))
    kk = vectorfind(s,PT(:,r),'c')
    if isempty(kk)
        conjunto_teste = [conjunto_teste PT(:,r)]
        P_TESTE = conjunto_teste(1:2,:)
        T_TESTE = conjunto_teste(3,:)        
    end
end

//---------------------------------------------

// DEFINIÇÃO DE PARÂMETROS DE TREINAMENTO 
// PARTES DO CÓDIDO FORAM COMENTADAS POIS ERAM RELACIONADAS À FUNÇÃO ann_FFBP_lm


        mu = 0.000001; // TAXA DE APRENDIZADO

        mumax = 100000//00000; // TAXA DE APRENDIZADO MÁXIMA

        theta = 5;     // FATOR DE AJUSTE DA TAXA DE APRENDIZADO

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
teste = ann_FFBP_run(P_TESTE,W,af);
testex = T_TESTE;
figure(103);
plot(testex,teste);
plot(testex,testex,'r')
legend('Saída da Rede', 'Saída Esperada')

