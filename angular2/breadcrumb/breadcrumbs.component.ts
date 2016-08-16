/*
 * Angular 2 decorators and services
 */
import { Component, OnInit, OnDestroy } from '@angular/core';

import { BreadcrumbComponent } from './breadcrumb';
import { BreadcrumbsService, BreadcrumbModel } from '../../services/breadcrumbs';
import { Observable } from 'rxjs/Rx';



/**
 * This component define the breadcrumbs container to hold the breadcrumbs
 * This can be devided further to a sub-component like breadcrumb-item.
 *
 */
@Component({
  selector: 'lb-breadcrumbs',
  pipes: [],
  providers: [],
  directives: [BreadcrumbComponent],
  templateUrl: './breadcrumbs.html'
})
export class BreadcrumbsComponent implements OnInit, OnDestroy {
  public crumbs: Observable<BreadcrumbModel[]>;
  public pageType: string = '';
  public pageTitle: string = '';
  public pageNumber: string = '';
  public errorMessage: string;
  private _breadcrumbService: BreadcrumbsService;

  constructor(breadcrumbService: BreadcrumbsService) {
    this._breadcrumbService = breadcrumbService;
    this.crumbs = this._breadcrumbService.breadcrumbsReadySubscription;
  }

  /**
   * ngOnInit - it will subscribe to visibility emitter
   */
  public ngOnInit(): void {
  }
  /**
   * To catch navChangeSubscription event from Navigation Service
   */

  public ngOnDestroy(): void {
  }


}
