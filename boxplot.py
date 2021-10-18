import pandas as pd
import matplotlib.pyplot as plt  # 导入图像库
import matplotlib.ticker as ticker

def Boxplot(tre_all):

    # diff = diff.numpy()


    # mean1 = tre1.mean().numpy()
    # std1 = tre1.std().numpy()

    dt = pd.DataFrame(tre_all)







    # plt.xticks([])
    plt.xlabel("", fontsize=16)

    plt.ylabel('error', fontsize=16)
    # plt.grid(linestyle="--", alpha=0.8)
    # plt.title('case%d'%case)


    dt.boxplot(grid=False)  # 画箱线图，直接使用DataFrame的方法



    # for axis in [ax.xaxis, ax.yaxis]:
    #     axis.set_major_locator(ticker.MaxNLocator(integer=True))
    # plt.yticks(fontsize=5)
    plt.show()