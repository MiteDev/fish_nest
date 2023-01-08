import { Controller, Get, Post } from '@nestjs/common';
import { FishService } from './fish.service';

@Controller('fish')
export class FishController {
    constructor( private fishService: FishService) {}

    @Get()
    async dataList() {
        
    }

    @Get('download')
    async dataDownload() {

    }

    @Get('imgdown')
    async imgDownload() {

    }
    

    @Post('upload')
    async uploadData() {

    }
}
