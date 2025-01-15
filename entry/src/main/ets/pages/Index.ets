import { hilog } from '@kit.PerformanceAnalysisKit';
import testNapi from 'libentry.so';
import { picker } from '@kit.CoreFileKit';
import { fileIo as fs } from '@kit.CoreFileKit';
import { common, UIAbility, Want } from '@kit.AbilityKit';
import { BusinessError, emitter } from '@kit.BasicServicesKit';
import { taskpool } from '@kit.ArkTS';
import featureAbility from '@ohos.ability.featureAbility';
import { promptAction } from '@kit.ArkUI';

const TAG: string = '[Sample_FAModelAbilityDevelop]';
const domain: number = 0xFF00;

let want: Want = {
  deviceId: "",
  bundleName: "com.aloereed.myapplication",
  abilityName: "ServerAbility",
};

function uriToPath(uri: string): string {
  if (!uri.startsWith('file://')) {
    throw new Error('The provided URI is not a file URI.');
  }

  let path = uri.slice(7);

  if (path[0] === '/' && /^[a-zA-Z]:/.test(path.slice(1, 3))) {
    path = path.slice(1);
  }

  if (path.startsWith('docs/')) {
    path = path.slice(4);
  } else if (path.startsWith('media/')) {
    path = path.slice(5);
  }

  path = decodeURIComponent(path);

  return path;
}

@Concurrent
async function StartServer(command: string): Promise<void> {
  emitter.on({ eventId: 1 }, () => {
    emitter.off(1)
  });
  testNapi.openllama(command);
}


@Entry
@Component
struct Index {
  @State message: string = 'Hello World';
  @State selectedModel: string = '没有模型被选择。';
  @State contextSize: number = 4096;
  @State port: number = 8080;
  @State otherargs: string = '';
  private task1: taskpool.Task = new taskpool.Task(StartServer, '');
  private context = getContext(this) as common.UIAbilityContext;
  async toWebBrowser(url: string): Promise<void> {
    // 获取当前应用的UIAbility上下文

    // 创建一个Want对象，用于描述启动系统浏览器的意图
    let want: Want = {
      // 指定动作类型，这里使用的是查看数据的动作
      action: 'ohos.want.action.viewData',
      // 指定实体类型，这里表示可浏览的实体
      entities: ['entity.system.browsable'],
      // 设置要打开的网页地址
      uri: url,
      // 设置额外的参数，这里包含了拉起方应用的包名
      parameters: {
        // 应用包名，用于标识拉起方应用
        bundleName: this.context.abilityInfo.bundleName,
        // 传递给浏览器的参数，这里同样使用了应用包名
        pushParams: this.context.abilityInfo.bundleName,
      }
    };
    // 使用UIAbility上下文启动系统浏览器，传入之前创建的Want对象
    return await this.context.startAbility(want);
  }
  build() {
    Row() {
      Scroll(){
      Column() {
        Text('Llama.cpp 服务器配置')
          .fontSize(24)
          .fontWeight(FontWeight.Bold)
          .margin({ bottom: 20 });

        Text('已选择的模型: ' + this.selectedModel)
          .fontSize(18)
          .margin({ bottom: 20 });

        TextInput({ placeholder: '提示词上下文大小 (默认: 4096)' })
          .width('80%')
          .height(40)
          .margin({ bottom: 20 })
          .onChange((value: string) => {
            this.contextSize = parseInt(value) || 4096;
          });

        TextInput({ placeholder: '端口 (默认: 8080)' })
          .width('80%')
          .height(40)
          .margin({ bottom: 20 })
          .onChange((value: string) => {
            this.port = parseInt(value) || 8080;
          });

        TextInput({ placeholder: '其他命令行参数 (默认为空)' })
          .width('80%')
          .height(40)
          .margin({ bottom: 20 })
          .onChange((value: string) => {
            this.otherargs = value || '';
          });

        Button("选择 GGUF 模型")
          .width('80%')
          .height(40)
          .margin({ bottom: 20 })
          .onClick(() => {
            const documentSelectOptions = new picker.DocumentSelectOptions();
            documentSelectOptions.maxSelectNumber = 1;
            documentSelectOptions.fileSuffixFilters = ['.gguf'];
            documentSelectOptions.authMode = false;

            let context = getContext(this) as common.Context;
            const documentViewPicker = new picker.DocumentViewPicker(context);

            documentViewPicker.select(documentSelectOptions).then((documentSelectResult: Array<string>) => {
              this.selectedModel = uriToPath(documentSelectResult[0]);
              console.info('Selected model: ' + this.selectedModel);
            }).catch((err: BusinessError) => {
              console.error(`Invoke documentViewPicker.select failed, code is ${err.code}, message is ${err.message}`);
            });
          });

        Button("启动服务器")
          .width('80%')
          .height(40)
          .margin({ bottom: 20 })
          .onClick(() => {
            let command = `./llama-cli -m ${this.selectedModel} --host 0.0.0.0 -c ${this.contextSize} --port ${this.port} ${this.otherargs}`;
            this.task1 = new taskpool.Task(StartServer, command);
            taskpool.execute(this.task1, taskpool.Priority.LOW);

            this.message = `服务器在端口 ${this.port} 上启动，窗口大小为 ${this.contextSize}`;
          });
        Button("关闭服务器")
          .width('80%')
          .height(40)
          .margin({ bottom: 20 })
          .onClick(() => {
            emitter.emit({ eventId: 1 });
            emitter.off(0);
            try {
              taskpool.terminateTask(this.task1);
            } catch (e) {
              console.error(e);
            }

            this.message = `请直接关闭本软件。`;
          });
        Text(this.message)
          .fontSize(16)
          .margin({ top: 20 });

        Text('请使用Chatbox、OpenWebUI等工具连接下面的地址:')
          .fontSize(16)
          .margin({ top: 20 });

        Text(`http://你的设备IP（本机为127.0.0.1）:${this.port}`)
          .fontSize(16)
          .fontColor(Color.Blue)
          .margin({ top: 10 });
        Text(`去ModelScope下载通义千问模型`)
          .fontSize(16)
          .fontColor(Color.Blue)
          .margin({ top: 10 }).onClick(() => {
            this.toWebBrowser('https://www.modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct-GGUF/files');
          });
        Text(`Llama.cpp 命令行参数一览`)
          .fontSize(16)
          .fontColor(Color.Blue)
          .margin({ top: 10 }).onClick(() => {
          this.toWebBrowser('https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md');
        });
      }
      .width('100%')
      .padding(20)
    }
    }
    .height('100%')
    .backgroundColor(Color.White)
  }
}
