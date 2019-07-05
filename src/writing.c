#include "network.h"
#include "utils.h"
#include "parser.h"

#ifdef OPENCV
#include <opencv2/highgui/highgui_c.h>
#endif

void train_writing(char *cfgfile, char *weightfile);
void train_writing_gpu( char *datacfg , char *cfgfile, char *weightfile , int *gpus, int ngpus );
void test_writing(char *cfgfile, char *weightfile, char *filename);
float valid_writing( char *datacfg , char *cfgfile, char *weightfile , network *existing_net );

float valid_writing( char *datacfg , char *cfgfile, char *weightfile , network *existing_net )
{
    // network net = parse_network_cfg(cfgfile);
    // if(weightfile){
    //     load_weights(&net, weightfile);
    // }
    // set_batch_network(&net, 1);
    int i, j;
    network net;
    int old_batch = -1;
    if (existing_net) {
        net = *existing_net;    // for validation during training
        old_batch = net.batch;
        // set_batch_network(&net, 1);
    }
    else {
        net = parse_network_cfg_custom(cfgfile, 1, 0);
        if (weightfile) {
            load_weights(&net, weightfile);
        }
        //set_batch_network(&net, 1);
        // fuse_conv_batchnorm(net);
        // calculate_binary_weights(net);
    }
    srand(2222222);
    clock_t time;
    // char buff[256];
    printf("Reading cfg file\n");
    list *options = read_data_cfg(datacfg);
    char *valid_list = option_find_str(options, "valid", "figures_valid.list");
    list *plist = get_paths(valid_list);

    // list *plist = get_paths("figures_valid.list");
    int m = plist->size;

    char **paths = (char **)list_to_array(plist);
    // printf("Replacing names\n");
    char **replace_paths = find_replace_paths(paths, m, ".jpg", "_label.jpg");
    
    float avg_iou = 0;

    free_list(plist);
    printf("Start predicting\n");
    for(i = 0; i < m; ++i){
        image im = load_image_color(paths[i], 0, 0);
        image resized = resize_min(im, net.w);
        image crop = crop_image(resized, (resized.w - net.w)/2, (resized.h - net.h)/2, net.w, net.h);
        // printf("predicting...\n");
        network_predict(net, crop.data);
        // printf("get result\n");
        image pred = get_network_image(net);//mirror of final layer
        // printf("tresholding\n");
        image thresh = threshold_image(pred, .5);
        // printf("loading label\n");
        image label = load_image(replace_paths[i], thresh.w, thresh.h, 3);
        // printf("grascaling\n");
        image labelgray = grayscale_image(label);
        // image bingray = binarize_image(labelgray);
        // printf("iou\n");
        float iou = iou_binary_image(thresh,labelgray);
        avg_iou += iou;
        // printf("cout\n");
        // if (existing_net) printf("\r");
        // else printf("\n");
        if ( i%1000 == 0)
            printf("%d: avg_iou = %f\n", i , avg_iou/i  );
        // printf("free data\n");
        if(resized.data != im.data) free_image(resized);
        free_image(crop);
        free_image(im);
        free_image(thresh);
        // free_image(pred);
        free_image(label);
        free_image(labelgray);


    }//for loop

    if (existing_net) {
        set_batch_network(&net, old_batch);
    } else {
        free_network(net);
    }

    for(i = 0; i < m; ++i) {
        free(replace_paths[i]);
        free(paths[i]);
    }
    free(replace_paths);
    free(paths);
    free_list(options);

    printf("Avg_iou = %f\n",  avg_iou / m );

    return avg_iou / m;

}


void train_writing(char *cfgfile, char *weightfile)
{
    int i,j;
    char* backup_directory = "backup/writing/";
    srand(time(0));
    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = net.batch*net.subdivisions;
    list *plist = get_paths("figures.list");
    char **paths = (char **)list_to_array(plist);
    clock_t time;
    int N = plist->size;
    printf("N: %d\n", N);
    image out = get_network_image(net);

    free_list(plist);

    data train, buffer;

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.out_w = out.w;
    args.out_h = out.h;
    args.paths = paths;
    args.n = imgs;
    args.m = N;
    args.d = &buffer;
    args.type = WRITING_DATA;

    pthread_t load_thread = load_data_in_thread(args);
    int epoch = (*net.seen)/N;
    while(get_current_batch(net) < net.max_batches || net.max_batches == 0){
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_in_thread(args);
        printf("Loaded %lf seconds\n",sec(clock()-time));

        time=clock();
        float loss = train_network(net, train);
        // float loss = 0;
        // #ifdef GPU
        //         if (ngpus == 1) {
        //             loss = train_network(net, train);
        //         }
        //         else {
        //             loss = train_networks(nets, ngpus, train, 4);
        //         }
        // #else
        //         loss = train_network(net, train);
        // #endif

        /*
           image pred = float_to_image(64, 64, 1, out);
           print_image(pred);
         */

        /*
           image im = float_to_image(256, 256, 3, train.X.vals[0]);
           image lab = float_to_image(64, 64, 1, train.y.vals[0]);
           image pred = float_to_image(64, 64, 1, out);
           show_image(im, "image");
           show_image(lab, "label");
           print_image(lab);
           show_image(pred, "pred");
           cvWaitKey(0);
         */

        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%d, %.3f: %f, %f avg, %f rate, %lf seconds, %ld images\n", get_current_batch(net), (float)(*net.seen)/N, loss, avg_loss, get_current_rate(net), sec(clock()-time), *net.seen);
        free_data(train);
        if(get_current_batch(net)%100 == 0){
            char buff[256];
            sprintf(buff, "%s/%s_batch_%d.weights", backup_directory, base, get_current_batch(net));
            save_weights(net, buff);
        }
        if(*net.seen/N > epoch){
            epoch = *net.seen/N;
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",backup_directory,base, epoch);
            save_weights(net, buff);
        }
    }

    for(i = 0; i < N; ++i) {
        free(paths[i]);
    }
    free(paths);
}

void train_writing_gpu(char *datacfg, char *cfgfile, char *weightfile , int *gpus, int ngpus )
{
    int i,j;
    char* backup_directory = "backup/writing/";
    srand(time(0));
    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    printf("%d\n", ngpus);

    // network net_map;
    // cuda_set_device(gpus[0]);
    // printf(" Prepare additional network for mAP calculation...\n");
    // net_map = parse_network_cfg_custom(cfgfile, 1, 1);


    // int k;  // free memory unnecessary arrays
    // for (k = 0; k < net_map.n; ++k) {
    //         free_layer(net_map.layers[k]);
    // }


    //20190705 +gpu
    network* nets = (network*)calloc(ngpus, sizeof(network));
    int seed = rand();
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = parse_network_cfg(cfgfile);
        if(weightfile){
            load_weights(&nets[i], weightfile);
        }
        // if(clear) *nets[i].seen = 0;
        nets[i].learning_rate *= ngpus;
    }
    srand(time(0));
    //+gpu
    network net = nets[0];



    // network net = parse_network_cfg(cfgfile);
    // if(weightfile){
    //     load_weights(&net, weightfile);
    // }

    list *options = read_data_cfg(datacfg);
    char *train_list = option_find_str(options , "train" , "figures.list");
    list *plist = get_paths(train_list);


    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = net.batch*net.subdivisions;
    // list *plist = get_paths("figures.list");
    char **paths = (char **)list_to_array(plist);
    clock_t time;
    int N = plist->size;
    printf("N: %d\n", N);
    image out = get_network_image(net);

    free_list(plist);

    data train, buffer;

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.out_w = out.w;
    args.out_h = out.h;
    args.paths = paths;
    args.n = imgs;
    args.m = N;
    args.d = &buffer;
    args.threads = 32;
    args.type = WRITING_DATA;

    args.threads = 3 * ngpus;

    //+gpu
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_data(args);
    // pthread_t load_thread = load_data_in_thread(args);

    int epoch = (*net.seen)/N;
    while(get_current_batch(net) < net.max_batches || net.max_batches == 0){
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);
        // load_thread = load_data_in_thread(args);
        printf("Loaded %lf seconds\n",sec(clock()-time));

        time=clock();
        // float loss = train_network(net, train);
        float loss = 0;
#ifdef GPU
        if (ngpus == 1) {
            loss = train_network(net, train);
        }
        else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif

        i = get_current_batch(net);
        if (i % 10 == 0) {
            if (net.cudnn_half) {
                if (i < net.burn_in * 3) fprintf(stderr, " Tensor Cores are disabled until the first %d iterations are reached.\n", 3 * net.burn_in);
                else fprintf(stderr, " Tensor Cores are used.\n");
            }
        }

        // if (i%100 == 0) {
        //     copy_weights_net(net, &net_map);
        //     printf("##validation##\n");
        //     float avg_iou = valid_writing(datacfg , cfgfile, weightfile , &net_map );
        //     printf("%d: avg_iou = %f\n",  i , avg_iou );
        // }


        /*
           image pred = float_to_image(64, 64, 1, out);
           print_image(pred);
         */

        /*
           image im = float_to_image(256, 256, 3, train.X.vals[0]);
           image lab = float_to_image(64, 64, 1, train.y.vals[0]);
           image pred = float_to_image(64, 64, 1, out);
           show_image(im, "image");
           show_image(lab, "label");
           print_image(lab);
           show_image(pred, "pred");
           cvWaitKey(0);
         */

        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%d, %.3f: %f, %f avg, %f rate, %lf seconds, %ld images\n", get_current_batch(net), (float)(*net.seen)/N, loss, avg_loss, get_current_rate(net), sec(clock()-time), *net.seen);
        // free_data(train);
        if(get_current_batch(net)%100 == 0){
#ifdef GPU
            if (ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s_batch_%d.weights", backup_directory, base, get_current_batch(net));
            save_weights(net, buff);
        }
//         if(*net.seen/N > epoch){
//             epoch = *net.seen/N;
// #ifdef GPU
//             if (ngpus != 1) sync_nets(nets, ngpus, 0);
// #endif
//             char buff[256];
//             sprintf(buff, "%s/%s_%d.weights",backup_directory,base, epoch);
//             save_weights(net, buff);
//         }
        free_data(train);
    }//while
    free_network(net);

    // net_map.n = 0;
    // free_network(net_map);

    for(i = 0; i < N; ++i) {
        free(paths[i]);
    }
    free(paths);
    free_list(options);
}

void test_writing(char *cfgfile, char *weightfile, char *filename)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;
    char buff[256];
    char *input = buff;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        }else{
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }

        image im = load_image_color(input, 0, 0);
        resize_network(&net, im.w, im.h);
        printf("%d %d %d\n", im.h, im.w, im.c);
        float *X = im.data;
        time=clock();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        image pred = get_network_image(net);

        image upsampled = resize_image(pred, im.w, im.h);
        image thresh = threshold_image(upsampled, .5);
        free_image(pred);//++ M em Leak
        pred = thresh;

        //donot show just save
        // show_image(pred, "prediction");
        // show_image(im, "orig");
        save_image(pred, "predictions");
#ifdef OPENCV
//         cvWaitKey(0);
        cvDestroyAllWindows();
#endif

        free_image(upsampled);
        free_image(thresh);
        free_image(im);
        // free_image(pred); because pred = thresh
        if (filename) break;
    }
}

void run_writing(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)][datacfg]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5] : 0;//imagename / datacfg file name

    //20190705 +gpu
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if(gpu_list){
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = (int*)calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }
    //+gpu




    if(0==strcmp(argv[2], "train")) train_writing(cfg, weights);
    else if(0==strcmp(argv[2], "traingpu")) train_writing_gpu( filename, cfg, weights , gpus , ngpus );
    else if(0==strcmp(argv[2], "test")) test_writing(cfg, weights, filename);
    else if(0==strcmp(argv[2], "valid")) valid_writing( filename, cfg, weights,NULL);
}
